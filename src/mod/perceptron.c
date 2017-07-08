/**
* @file perceptron.c
* @author P Mitrik
* @date 22 Jun 2017
* @copyright 2017 P Mitrik
* @brief The functions to perform the perceptron algorithm.
*
*/
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/random.h>


static s64 *pWeights;

// Set some default values
static s64 learningRate        = 100; // 0.1 * 1000
static u32 inputs              = 1;
static u32 outputs             = 1;

/**
* @brief Set the number of input neurons
* @param [in] neurons the number of neurons for the input layer
* @details Sets the input layer of neurons
*/
void setInputNeurons(const u32 neurons) {
  inputs = neurons;
}

/**
* @brief Set the number of output neurons
* @param [in] neurons the number of neurons for the output layer
* @details Sets the output layer of neurons
*/
void setOutputNeurons(const u32 neurons) {
  outputs = neurons;
}

/**
* @brief Set the learning rate value.
* @param [in] learningRateValue the threshold to use
* @details this sets the learning rate to use throughout training
*/
void setlearningRate(const s64 learningRateValue) {
  learningRate = learningRateValue;
}

/**
* @brief Initialize the weights.
* @param [in] inputs the number of input neurons (Fa)
* @param [in] outputs the number of output neurons (Fb)
* @details Initialize the inter-layer connections and threshold values to values between [1, -1].
*/
void initializeWeights(void) {
  unsigned int row;
  unsigned int column;
  s64 i;

  // Create and allocate the inner weights
  if (NULL != pWeights) {
    kfree(pWeights);
  }
  pWeights = kcalloc(inputs * outputs, sizeof(s64), GFP_USER);

  // Set values belween [1, -1]. May also set to 0.0 as well.
  for (row = 0; row < inputs; row++) {
    for (column = 0; column < outputs; column++) {
        get_random_bytes(&i, sizeof(s64));
        *(pWeights + (row * outputs) + column) = (i % 1000) * 2 - 1000;
    }
  }
}

/**
* @brief Sum neuron values for the passed input values
* @param [in] pInputValues array of input values
* @return the bipolar results based upon the calculated sum
* @details Get the bipolar results from the summed input and weight values where sum < 0 is -1 otherwise 1.
*/
s32 * calculateActivationValue(const s32 *pInputValues) {
  s32 *result = kcalloc(inputs, sizeof(int), GFP_USER);
  s64 sum;
  u32 row;
  u32 column;

  if (NULL != pInputValues) {
    if (NULL != pWeights) {
      for (column = 0; column < outputs; column++) {
        sum = 0;
        for (row = 0; row < inputs; row++) {
          sum += (*(pWeights + (row * outputs) + column) * pInputValues[row]);
        }

        // Final value is a bipolar step function with values of 1 or -1
        result[column] = (((sum / 1000) < 0) ? -1 : 1);
      }
    }
  }

  return result;
}

/**
* @brief Training function for the current input.
* @param [in] pInputValues pointer to the input values
* @param [in] pActualOutput pointer to the actual output values
* @param [in] pDesiredOutput pointer to the desired output values
* @return Zero if training is finished otherwise some non-zero value
* @details Using the actual output and input, compare result to the desired outcome and make weight adjustments until training is zero.
*/
u32 trainingOutput(const s32 *pInputValues, const s32 *pActualOutput, const s32 *pDesiredOutput) {
  u32 row;
  u32 column;
  u32 training;
  s64 delta;

  // value used to determine if algorithm is still training
  training = 0;

  if (NULL != pInputValues) {
    if (NULL != pActualOutput) {
      if (NULL != pDesiredOutput) {
        if (NULL != pWeights) {
          for (column = 0; column < outputs; column++) {
            for (row = 0; row < inputs; row++) {
              delta = learningRate * pInputValues[row] * (pDesiredOutput[column] - pActualOutput[column]);

              // Adjust if the delta is not zero
              if (0 != delta) {
                *(pWeights + (row * outputs) + column) += delta;
//                printk(KERN_INFO "Perceptron: weight = %lld\n", *(pWeights + (row * outputs) + column));
                training++;
              }
            }
          }
        }
      }
    }
  }

  return training;
}

void finish(void) {
  if (NULL != pWeights) {
    kfree(pWeights);
  }
}
