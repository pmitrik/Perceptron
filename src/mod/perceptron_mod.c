#include <linux/module.h>
#include <linux/moduleparam.h>

#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <asm/uaccess.h>
#include <linux/mutex.h>

#include <linux/slab.h>
#include <linux/string.h>

#include "perceptron.h"

// These are parameters found under /sys/module/perceptron_module/parameters
// Information of the parameters can be found using command line 'modinfo <module_name>.ko'

static int input_neurons = 7;                                     // An example LKM argument
module_param_named(i_neurons, input_neurons, int, 0444);          // read only
MODULE_PARM_DESC(i_neurons, "The input neurons to use, max 10");  // parameter description

static int output_neurons = 6;                                    // An example LKM argument
module_param_named(o_neurons, output_neurons, int, 0444);         // read only
MODULE_PARM_DESC(o_neurons, "The output neurons to use, max 10"); // parameter description

static int outputArray[10] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
static int arr_argc = 0;
module_param_array(outputArray, int, &arr_argc, 0444);            // read only
MODULE_PARM_DESC(outputArray, "An array of integers");            // parameter description

static int inputArray[10] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
static int arr_argd = 0;
module_param_array(inputArray, int, &arr_argd, 0664);             // read/write for root/group
MODULE_PARM_DESC(inputArray, "An array of integers");             // parameter description

static DEFINE_MUTEX(perceptron_mutex);

static s32 *pActualResult;
static u32 training;
static u32 trainingSets;
static u32 loops;

static u32 input;
static u32 output;

// Some training data
static s32 inputValues[28][7] = {
  {-1, -1, -1, -1, -1, -1, -1}, // 0
  {-1, -1, -1, -1, -1, -1,  1}, // 1
  {-1, -1, -1, -1, -1,  1, -1}, // 2
  {-1, -1, -1, -1, -1,  1,  1}, // 3
  {-1, -1, -1, -1,  1, -1, -1}, // 4
  {-1, -1, -1, -1,  1, -1,  1}, // 5
  {-1, -1, -1, -1,  1,  1, -1}, // 6
  {-1, -1, -1, -1,  1,  1,  1}, // 7
  {-1, -1, -1,  1, -1, -1, -1}, // 8
  {-1, -1, -1,  1, -1, -1,  1}, // 9
  {-1, -1, -1,  1, -1,  1, -1}, // 10
  {-1, -1, -1,  1, -1,  1,  1}, // 11
  {-1, -1,  1, -1, -1, -1, -1}, // 16
  {-1, -1,  1, -1, -1, -1,  1}, // 17
  {-1, -1,  1, -1, -1,  1, -1}, // 18
  {-1, -1,  1, -1, -1,  1,  1}, // 19
  {-1, -1,  1, -1,  1, -1, -1}, // 20
  {-1,  1, -1, -1, -1, -1, -1}, // 32
  {-1,  1, -1, -1, -1, -1,  1}, // 33
  {-1,  1, -1, -1, -1,  1, -1}, // 34
  {-1,  1,  1, -1, -1, -1, -1}, // 48
  {-1,  1,  1, -1, -1, -1,  1}, // 49
  {-1,  1,  1, -1, -1,  1, -1}, // 50
  {-1,  1,  1, -1, -1,  1,  1}, // 51
  {-1,  1,  1,  1,  1, -1, -1}, // 60
  {-1,  1,  1,  1,  1, -1,  1}, // 61
  {-1,  1,  1,  1,  1,  1, -1}, // 62
  {-1,  1,  1,  1,  1,  1,  1}, // 63
};
static s32 desiredValues[28][6] = {
  { 1, 1,  1,  1,  1,  1}, // 63
  { 1, 1,  1,  1,  1, -1}, // 62
  { 1, 1,  1,  1, -1,  1}, // 61
  { 1, 1,  1,  1, -1, -1}, // 60
  { 1, 1,  1, -1,  1,  1}, // 59
  { 1, 1,  1, -1,  1, -1}, // 58
  { 1, 1,  1, -1, -1,  1}, // 57
  { 1, 1,  1, -1, -1, -1}, // 56
  { 1, 1, -1,  1,  1,  1}, // 55
  { 1, 1, -1,  1,  1, -1}, // 54
  { 1,  1, -1,  1, -1,  1}, // 53
  { 1,  1, -1,  1, -1, -1}, // 52
  { 1, -1,  1,  1,  1,  1}, // 47
  { 1, -1,  1,  1,  1, -1}, // 46
  { 1, -1,  1,  1, -1,  1}, // 45
  { 1, -1,  1,  1, -1, -1}, // 44
  { 1, -1,  1, -1,  1,  1}, // 43
  { -1,  1,  1,  1,  1,  1}, // 31
  { -1,  1,  1,  1,  1, -1}, // 30
  { -1,  1,  1,  1, -1,  1}, // 29
  { -1, -1,  1,  1,  1,  1}, // 15
  { -1, -1,  1,  1,  1, -1}, // 14
  { -1, -1,  1,  1, -1,  1}, // 13
  { -1, -1,  1,  1, -1, -1}, // 12
  { -1, -1, -1, -1,  1,  1}, // 3
  { -1, -1, -1, -1,  1, -1}, // 2
  { -1, -1, -1, -1, -1,  1}, // 1
  { -1, -1, -1, -1, -1, -1}, // 0
};

/**
** @brief Convert string value to integer value.
*/
static s32 toString(const char *a) {
  s32 c;
  s32 sign;
  s32 offset;
  s32 number;

  // Initialize variables
  offset = 0;
  sign   = 1;
  number = 0;

  if ('-' == a[offset]) {
    sign = -1;
    offset++;
  }

  // Calculate number
  for (c = offset; (a[c] != '\0') && (a[c] >= '0') && (a[c] <= '9'); c++) {
    number = number * 10 + a[c] - '0';
    printk(KERN_INFO "Perceptron: number = %d  ---  a[c] = %d\n", number, a[c]);
  }

  if (-1 == sign) {
    number = -number;
  }

  return number;
}

// --- File operations ---

static int perceptron_open(struct inode * inode, struct file * file)
{
    printk(KERN_INFO "Perceptron: open\n");

    return 0;
}

static int perceptron_release(struct inode * inode, struct file * file)
{
    printk(KERN_INFO "Perceptron: release\n");

    return 0;
}

/**
** @brief Copy outputArray from kernel space to user space
*/
static ssize_t perceptron_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
	char xyz_buf[100];
	int rc;

	printk("Perceptron: read\n");

	rc = copy_to_user(buf, xyz_buf, count);
	if(rc < 0) {
		printk(KERN_INFO "Perceptron: copy to user failed in read\n");
	}
    printk(KERN_INFO "Perceptron: read count = %d\n", (int) count);

	return count;
}

/**
** @brief Take the input from /dev/perceptron and run it through the trained algorithm.
*/
static ssize_t perceptron_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    char *ptr;
    char *buffer;
	char unused_buf[100];
    s32 number[10];
    u32 column;

//    char *space = "space\0";
//    char *null  = "NULL\0";

	printk(KERN_INFO "Perceptron: write\n");

	if (copy_from_user(unused_buf, buf, count)) {
		return -EFAULT;
	}
//    printk(KERN_INFO "Perceptron: write count = %d\n", (int) count);

    buffer = kmalloc(sizeof(unused_buf), GFP_USER);
    strcpy(buffer, unused_buf);
    column = 0;
    do {
        ptr = strsep(&buffer, ",");

//        printk(KERN_INFO "Perceptron: buffer = %s\n", buffer);
//        printk(KERN_INFO "Perceptron: ptr = %s\n", (' ' != *ptr) ? ((NULL != ptr) ? ptr : null) : space);

        if ((NULL != ptr) && (' ' != *ptr)) {
            number[column] = toString(ptr);
//            printk(KERN_INFO "Perceptron: number = %d\n", number[column]);

            column++;
        }
    } while ((NULL != buffer) && (column < input));
    kfree(buffer);

    // Run array through trained algorithm
    pActualResult = calculateActivationValue(&number[0]);
    for (column = 0; column < output; column++) {
        outputArray[column] = pActualResult[column];
//        printk(KERN_INFO "Perceptron: result[%d] = %d\n", column, pActualResult[column]);
    }

	return count;
}

// --- End file operations ---

static const struct file_operations perceptron_fops = {
    .owner    = THIS_MODULE,
    .open     = perceptron_open,
    .release  = perceptron_release,
    .write    = perceptron_write,
    .read     = perceptron_read,
};

// Create /dev/perceptron that is read/write for all groups
static struct miscdevice perceptron_dev = {
    .minor    = USERIO_MINOR,
    .name     = "perceptron",
    .fops     = &perceptron_fops,
    .mode     = 0666,
};

/* Loadable kernel module init functions*/
static int __init module_init_callback(void)
{
	int retval = 0;

    printk(KERN_ALERT "Perceptron: Initializing driver\n");

    mutex_lock(&perceptron_mutex);

    retval = misc_register(&perceptron_dev);
    if (!retval) {

        printk(KERN_ALERT "Perceptron: start training\n");

        input    = input_neurons;
        output   = output_neurons;
        arr_argc = output;

        setInputNeurons(input);
        setOutputNeurons(output);

        setlearningRate(100); // 0.1 * 1000
        initializeWeights();

        loops = 0;
        trainingSets = 10;
        do {
            training = 0;

            for (trainingSets = 0; trainingSets < 10; trainingSets++) {
              pActualResult = calculateActivationValue(&inputValues[trainingSets][0]);
              training += trainingOutput(&inputValues[trainingSets][0], &pActualResult[0], &desiredValues[trainingSets][0]);
            }
            loops++;
            printk(KERN_INFO "Perceptron: training set loop %d finished!!!\n", loops);
        } while (training);

        printk(KERN_INFO "Perceptron: training finished!!!\n");
    }

	return retval;
}

static void __exit module_exit_callback(void)
{
    finish();
    misc_deregister(&perceptron_dev);
    mutex_unlock(&perceptron_mutex);

	printk(KERN_ALERT "Perceptron: Exiting driver\n");
}

MODULE_DESCRIPTION("Perceptron driver");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0.0");

/** @brief A module must use the module_init() module_exit() macros from linux/init.h, which
 *  identify the initialization function at insertion time and the cleanup function (as
 *  listed above)
 */
module_init(module_init_callback);
module_exit(module_exit_callback);
