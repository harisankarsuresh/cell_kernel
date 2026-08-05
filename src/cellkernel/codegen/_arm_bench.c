/*
 * Bare-metal benchmark harness for a generated estimator.
 *
 * Runs a fixed number of filter steps on an emulated Cortex-M, then exits
 * through semihosting so the host can read QEMU's instruction count. There is no
 * operating system, no heap and no input: the whole point is that the work done
 * between the two reported counts is exactly CK_BENCH_STEPS calls to ck_step and
 * nothing else.
 *
 * The current profile is generated arithmetically rather than read from a file,
 * because file input would put library code in the measurement. It alternates
 * sign so the filter sees a realistic mix of charge and discharge rather than a
 * monotone ramp that might let a branch predictor settle into a pattern the real
 * duty cycle would not produce.
 */

#include "cellkernel_estimator.h"

#ifndef CK_BENCH_STEPS
#define CK_BENCH_STEPS 1000
#endif

/* ARM semihosting. Two calls are used: SYS_WRITE0 to report the count and
 * SYS_EXIT to hand control back so the host does not have to time out. */
static long semihost(int op, void *arg)
{
    register int r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = arg;
    register long result __asm__("r0");
    __asm__ volatile("bkpt #0xAB" : "=r"(result) : "r"(r0), "r"(r1) : "memory");
    return result;
}

static void semihost_write(const char *text)
{
    (void) semihost(0x04, (void *) (unsigned long) text); /* SYS_WRITE0 */
}

__attribute__((noreturn)) static void semihost_exit(void)
{
    unsigned long block[2] = {0x20026UL, 0UL}; /* ADP_Stopped_ApplicationExit */
    (void) semihost(0x18, block);              /* SYS_EXIT                    */
    for (;;) {
    }
}

/*
 * SysTick, used as a free-running 24-bit down-counter.
 *
 * The obvious instrument would be the data watchpoint unit's cycle counter, but
 * QEMU's Cortex-M boards do not implement it -- it reads zero. SysTick they do
 * implement, and under ``-icount shift=0`` QEMU advances virtual time by exactly
 * one nanosecond per instruction, so SysTick ticks are a faithful measure of
 * instructions retired: the board clock is 25 MHz, hence 40 instructions per
 * tick. The host divides that back out.
 */
#define SYST_CSR  (*(volatile unsigned long *) 0xE000E010UL)
#define SYST_RVR  (*(volatile unsigned long *) 0xE000E014UL)
#define SYST_CVR  (*(volatile unsigned long *) 0xE000E018UL)

static void counter_start(void)
{
    SYST_CSR = 0UL;
    SYST_RVR = 0x00FFFFFFUL;
    SYST_CVR = 0UL;
    SYST_CSR = 5UL; /* enable, processor clock, no interrupt */
}

/* Counts down, so elapsed is before minus after, wrapping in 24 bits. */
static unsigned long counter_read(void)
{
    return SYST_CVR & 0x00FFFFFFUL;
}

static void write_unsigned(const char *label, unsigned long value)
{
    char buffer[32];
    int i = (int) sizeof(buffer) - 1;
    buffer[i--] = '\0';
    buffer[i--] = '\n';
    if (value == 0UL) {
        buffer[i--] = '0';
    }
    while (value > 0UL && i >= 0) {
        buffer[i--] = (char) ('0' + (int) (value % 10UL));
        value /= 10UL;
    }
    semihost_write(label);
    semihost_write(&buffer[i + 1]);
}

/* Kept out of main so the optimiser cannot hoist it into the reset path. */
__attribute__((noinline)) static ck_real_t bench(void)
{
    ck_estimator_t est;
    ck_real_t acc = (ck_real_t) 0;
    int k;

    ck_init(&est, (ck_real_t) 0.8);
    for (k = 0; k < CK_BENCH_STEPS; ++k) {
        /* A square wave at 2C, alternating every 64 samples. */
        const ck_real_t current = ((k / 64) & 1) ? (ck_real_t) -5.0 : (ck_real_t) 5.0;
        acc += ck_step(&est, current, (ck_real_t) 3.8);
    }
    return acc;
}

/* Volatile so the whole benchmark cannot be optimised away as unused. */
volatile ck_real_t ck_bench_result;

/*
 * Calibration: how many instructions is one SysTick tick worth?
 *
 * Rather than assume the board clock and QEMU's icount scaling, this measures
 * it. The loop below runs CK_CALIBRATE_NOPS single-cycle instructions per
 * iteration on top of some unknown loop overhead. Run it twice with different
 * nop counts and difference the results, and the overhead -- whatever it is --
 * cancels exactly, leaving a known instruction count against a measured tick
 * count. No disassembly and no datasheet required.
 */
#ifdef CK_CALIBRATE_NOPS
#define CK_STR2(x) #x
#define CK_STR(x) CK_STR2(x)

__attribute__((noinline)) static void calibrate(void)
{
    volatile int i;
    for (i = 0; i < 10000; ++i) {
        /* Emitted by the assembler's repeat directive, so the count is exactly
         * CK_CALIBRATE_NOPS with no per-instruction loop overhead. A C loop here
         * would add its own increment, compare and branch to every nop and the
         * calibration would come out three times off. */
        __asm__ volatile(".rept " CK_STR(CK_CALIBRATE_NOPS) "\n\tnop\n\t.endr\n" ::: "memory");
    }
}
#endif

int main(void)
{
    unsigned long before, after;

    counter_start();
    before = counter_read();
#ifdef CK_CALIBRATE_NOPS
    calibrate();
#else
    ck_bench_result = bench();
#endif
    after = counter_read();

#ifdef CK_CALIBRATE_NOPS
    write_unsigned("CK_NOPS ", (unsigned long) CK_CALIBRATE_NOPS);
#else
    write_unsigned("CK_STEPS ", (unsigned long) CK_BENCH_STEPS);
#endif
    write_unsigned("CK_TICKS ", (before - after) & 0x00FFFFFFUL);
    semihost_exit();
}

/*
 * Minimal vector table and reset handler. Two entries is all QEMU needs to start
 * a Cortex-M image: the initial stack pointer and the reset vector.
 */
extern unsigned long _estack;

__attribute__((noreturn)) void Reset_Handler(void)
{
#if defined(__ARM_FP) && (__ARM_FP != 0)
    /* The floating-point unit is disabled out of reset on a Cortex-M4F, and the
     * first VFP instruction then raises a UsageFault that escalates straight to
     * lockup. Grant full access to coprocessors 10 and 11 before anything else
     * runs. Omitted on cores without an FPU, where CPACR does not exist. */
    volatile unsigned long *const cpacr = (volatile unsigned long *) 0xE000ED88UL;
    *cpacr |= (0xFUL << 20);
    __asm__ volatile("dsb" ::: "memory");
    __asm__ volatile("isb" ::: "memory");
#endif
    (void) main();
    for (;;) {
    }
}

__attribute__((section(".isr_vector"), used))
void *const ck_vectors[2] = {(void *) &_estack, (void *) Reset_Handler};
