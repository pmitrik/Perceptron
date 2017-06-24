/**
* @file test.c
* @author P Mitrik
* @date 22 Jun 2017
* @copyright 2017 P Mitrik
* @brief Example of how to use the perceptron functions.
*
*/
#include <stdio.h>
#include <stdlib.h>

#include "perceptron.h"

/*
** Hard-coded values for this example only. This example is doing a sequence
** recognition.
*/
int main(int argc, char const *argv[]) {
  int32_t *pActualResult;
  uint32_t column;
  uint32_t training;
  uint32_t trainingSets;
  uint32_t loops;
  uint32_t outputs;

  setInputNeurons(6);
  setOutputNeurons(4);

  outputs = 4;

  setlearningRate(0.1);
  initializeWeights();

  int32_t inputValues[10][6] = {
    {-1, -1, -1, -1, -1, -1}, // 0
    {-1, -1, -1, -1, -1,  1}, // 1
    {-1, -1, -1, -1,  1, -1}, // 2
    {-1, -1, -1, -1,  1,  1}, // 3
    {-1, -1, -1,  1, -1, -1}, // 4
    {-1, -1, -1,  1, -1,  1}, // 5
    {-1, -1, -1,  1,  1, -1}, // 6
    {-1, -1, -1,  1,  1,  1}, // 7
    {-1, -1,  1, -1, -1, -1}, // 8
    {-1, -1,  1, -1, -1,  1}, // 9
  };
  int32_t desiredValues[10][4] = {
    { 1,  1,  1,  1}, // 15
    { 1,  1,  1, -1}, // 14
    { 1,  1, -1,  1}, // 13
    { 1,  1, -1, -1}, // 12
    { 1, -1,  1,  1}, // 11
    { 1, -1,  1, -1}, // 10
    { 1, -1, -1,  1}, // 9
    { 1, -1, -1, -1}, // 8
    {-1,  1,  1,  1}, // 7
    {-1,  1,  1, -1}, // 6
  };

  loops = 0;
  trainingSets = 10;
  do {
      training = 0;

      for (trainingSets = 0; trainingSets < 10; trainingSets++) {
        pActualResult = calculateActivationValue(&inputValues[trainingSets][0]);
        training += trainingOutput(&inputValues[trainingSets][0], &pActualResult[0], &desiredValues[trainingSets][0]);
      }
      loops++;
      printf("Training set loop %d finished!!!\n", loops);
  } while (training);

  if (NULL != pActualResult) {
    free(pActualResult);
  }
  printf("Training finished!!!\n\n\n");

  // Test input {1, 1, 1, 1, 1, -1}, expect result of {-1, -1, -1, 1}
  int32_t values[] = {1, 1, 1, 1, 1, -1};

  pActualResult = calculateActivationValue(&values[0]);
  for (column = 0; column < outputs; column++) {
    printf("final result[%d]: %d\n", column, pActualResult[column]);
  }

  if (NULL != pActualResult) {
    free(pActualResult);
  }
  finish();

  /* code */
  return 0;
}
