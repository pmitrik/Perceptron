/**
* @file perceptron.h
* @author P Mitrik
* @date 22 Jun 2017
* @copyright 2017 P Mitrik
* @brief The functions that are accessible to the outside world.
*
*/
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <linux/types.h>

#ifdef __cplusplus
extern "C" {
#endif

void setInputNeurons(const u32 neurons);
void setOutputNeurons(const u32 neurons);
void setlearningRate(const s64 learningRateValue);
void initializeWeights(void);
s32 * calculateActivationValue(const s32 *pInputValues);
u32 trainingOutput(const s32 *pInputValues, const s32 *pActualOutput, const s32 *pDesiredOutput);
void finish(void);

#ifdef __cplusplus
}
#endif

#endif
