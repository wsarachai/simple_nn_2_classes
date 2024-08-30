#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//Logistic regression training on uno
//cat dog prediction using height and weight
#define epoch 200000
#define n_features 2
#define samples 8
#define test_samples 5

float learning_rate = 0.05;

// Training set
float X[]={0.9,0.7,0.7,0.8,0.3,0.4,0.2,0.3,0.6,0.6,0.8,0.7,0.2,0.2,0.25,0.35};
float ytrue[] = {1,1,0,0,1,1,0,0};

// Test set
float test_X[]={0.2,0.1,0.55,0.65,0.231,0.475,0.15,0.373,0.65,0.76}; //test samples
float test_ytrue[] = {0,1,0,0,1};

float w[] = { 0.1, 0.1 };
float b = 0.2;
float tmp[2];
float dw1, dw2, db;

float y[samples];

// sigmoid activation is coded
float sigmoid(float x) {
  double y = 1/(1+exp(-x));
  return y;
}

//code for n-input neuron with sigmoid activation
float neuron(float W[], float B, float X[], int nn) // n is feature size
{
  int i;
  float z, y = 0.0;
  for(i=0; i < nn; i++) {
     y += W[i]*X[i];
  }
  y += B;
  z = sigmoid(y);
  return z;
}

//calculate cross entropy loss
float loss_crossentropy(float ytrue[], float ypred[], int num) {
  int i;
  float loss= 0;
  for(i=0;i<num;i++) {
    loss += -ytrue[i]*log(ypred[i])-(1-ytrue[i])*log(1-ypred[i]);
  }
  loss /= num; // ength of test data is 4
  return loss;
}

//Gradient Descent algorithm
void gradient_descent(float ytrue[], float ypred[]) {
  int i;
  //float loss= 0;
  dw1=0;
  dw2=0;
  db=0;
  for(i=0;i<samples;i++) {
    db += (ypred[i]-ytrue[i]);
    int jj = 2*i;
    dw1 += X[jj]*(ypred[i]-ytrue[i]);
    dw2 += X[jj+1]*(ypred[i]-ytrue[i]);
  }
  //for(int k=0;k<samples;k++) {
  // int jj = 2*k;
  //dw1 += X[jj]*db;
  //dw2 += X[jj+1]*db;
  //}
  db /= samples; //length of test data is 4
  dw1 /= samples;
  dw2 /= samples;
}

void train() {
  int i, n=0;
  printf("Training the model...");
  
  unsigned int t1 = clock();
  float t2 = 0;
  while(n <= epoch) { //start the training loop for num of epoch
    int j=0;
    for(i=0; i< samples; i++) {
      j = n_features*i;
      tmp[0] = X[j];
      tmp[1] = X[j+1];
      y[i] = neuron(w,b,tmp,n_features);
      //Serial.print("\n");
      // Serial.println(y[i],5);
      //j = i+2;
    }
    float loss = loss_crossentropy(ytrue, y, samples);
    float accuracy = 1-loss;
    printf("\nNumber of Epoch: ");
    printf("%d", n);
    printf("\nTraining Loss: ");
    printf("%.5f", loss);
    printf("\tAccuracy: ");
    printf("%.2f", accuracy);
    //call gradient descent
    gradient_descent(ytrue,y);
    
    if (n > 50000) learning_rate = 0.025;
	if (n > 100000) learning_rate = 0.005;
	    
    w[0] -= learning_rate * dw1;
    w[1] -= learning_rate * dw2;
    b -= learning_rate * db;
    n += 1;
    /*Serial.print("\n");
//    Serial.print(w[0],5);
//    Serial.print("\t");
//    Serial.print(w[1],5);
//    Serial.print("\n");
//    */
  }
  printf("\nTraining Completed Successfully... ");
  printf("\nPrinting Weights and bias values: w[0], w[1], b ==> ");
  printf("[%.5f, %.5f, %.5f]", w[0], w[1], b);
  t2 = (clock()-t1)/1000;
  printf("\nTime elapsed: ");
  printf("%.2f", t2);
  printf(" Sec");
}

void test() {
  //****** Prediction using evaluated model **//
  printf("\nPredicted values: ");
  int i, j=0;
  for(i=0; i< test_samples; i++) //4 is total training data len
  {
    j = n_features*i;
    tmp[0] = test_X[j];
    tmp[1] = test_X[j+1];
    y[i] = neuron(w,b,tmp,n_features);
    
    if(y[i]>0.5)
      printf("1, ");
    else
      printf("0, ");
  }
  float loss = loss_crossentropy(test_ytrue, y, test_samples);
  float accuracy = 1-loss;
  printf("\nAccuracy: %.2f", accuracy);
}

int main(int argc, char *argv[]) {
  train();
  test();
  return 0;
}
