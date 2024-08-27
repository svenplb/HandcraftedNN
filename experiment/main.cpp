#include <iostream>



using namespace std;

int classify(double input_1, double input_2);

// Weight values connecting each input to the first output
double weight_1_1, weight_2_1;
// Weight values connecting each input to the second output
double weight_1_2, weight_2_2;
// Bias | y = k*x+d (d)
double bias_1, bias_2;


int main(void) {
	double input_1 = 123.123;
	double input_2 = 3.12;
	int returnValue = classify(input_1, input_2);
	cout << returnValue;
}

int classify(double input_1, double input_2) {
	// f(x) = k*x+d | lienar function
	double output_1 = input_1 * weight_1_1 + input_2 * weight_2_1 + bias_1;
	double output_2 = input_2 * weight_1_2 + input_2 * weight_2_2 + bias_2;

	cout << output_1;
	cout << output_2;
	
	int returnVal;

	// output1 higher = safe
	// output2 higher = poison
	if(output_1 > output_2) {
		//safe
		return 0;
	}

	//poison
	return 1;
}
