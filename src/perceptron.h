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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void setInputNeurons(const int32_t neurons);
void setOutputNeurons(const int32_t neurons);
void setlearningRate(const double learningRateValue);
void initializeWeights();
int32_t * calculateActivationValue(const int32_t *pInputValues);
uint32_t trainingOutput(const int32_t *pInputValues, const int32_t *pActualOutput, const int32_t *pDesiredOutput);
void finish(void);

#ifdef __cplusplus
}
#endif

#endif
