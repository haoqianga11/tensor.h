#include "tensor.h"
#include <time.h>
#include <sys/time.h>

// 加载CSV文件的函数（与test.c相同）
void load_csv(Tensor* x, Tensor* y, char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(1);
    }

    char line[10000];
    char *token;
    
    for(int b = 0; b < 60000; b++) {
        if(fgets(line, sizeof(line), file) != NULL) {
            token = strtok(line, ",");
            for(int i = 0; i < 28*28 + 10; i++) {
                if (token == NULL) {
                    fprintf(stderr, "CSV format error: not enough columns\n");
                    fclose(file);
                    exit(1);
                }
                if(i < 28*28) {
                    x->data->values[b * 28 * 28 + i] = atof(token);
                } else {
                    y->data->values[b * 10 + (i - 28*28)] = atof(token) * (-1.0f);
                }
                token = strtok(NULL, ",");
            }
        } else {
            break;
        }
    }
    fclose(file);
}

// 获取预测的数字
int get_prediction(Tensor* output) {
    int pred = 0;
    float max_val = output->data->values[0];
    for(int i = 1; i < 10; i++) {
        if(output->data->values[i] > max_val) {
            max_val = output->data->values[i];
            pred = i;
        }
    }
    return pred;
}

// 获取真实标签
int get_true_label(Tensor* y, int index) {
    int label = 0;
    float min_val = y->data->values[index * 10];
    for(int i = 1; i < 10; i++) {
        if(y->data->values[index * 10 + i] < min_val) {
            min_val = y->data->values[index * 10 + i];
            label = i;
        }
    }
    return label;
}

// Print digit image
void print_digit(Tensor* x, int index) {
    printf("\nDigit Image:\n");
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            float pixel = x->data->values[index * 784 + i * 28 + j];
            if(pixel > 0.5) printf("##");
            else if(pixel > 0.2) printf("++");
            else if(pixel > 0.0) printf("..");
            else printf("  ");
        }
        printf("\n");
    }
}

int main() {
    // Load test data
    Tensor* x = create_zero_tensor((int[]){60000,784}, 2);
    Tensor* y = create_zero_tensor((int[]){60000,10}, 2);
    load_csv(x, y, "mnist_test.csv");
    
    // Create model weights
    Tensor* w1 = create_zero_tensor((int[]){784,128}, 2);
    Tensor* w2 = create_zero_tensor((int[]){128,10}, 2);
    
    // Load trained weights
    FILE *f1 = fopen("w1.weights", "rb");
    FILE *f2 = fopen("w2.weights", "rb");
    if(!f1 || !f2) {
        printf("Error: Weight files not found! Please run training program first.\n");
        return 1;
    }
    fread(w1->data->values, sizeof(float), w1->data->size, f1);
    fread(w2->data->values, sizeof(float), w2->data->size, f2);
    fclose(f1);
    fclose(f2);
    
    // Test first 10 samples
    int correct = 0;
    for(int i = 0; i < 10; i++) {
        // Create tensor for single sample
        Tensor* sample = create_zero_tensor((int[]){1,784}, 2);
        for(int j = 0; j < 784; j++) {
            sample->data->values[j] = x->data->values[i * 784 + j];
        }
        
        // Forward pass
        Tensor* w1_out = matmul(sample, w1);
        Tensor* relu_out = relu(w1_out);
        Tensor* w2_out = matmul(relu_out, w2);
        Tensor* output = logsoftmax(w2_out);
        
        // Get prediction
        int pred = get_prediction(output);
        int true_label = get_true_label(y, i);
        
        // Print results
        printf("\n===== Sample %d =====", i+1);
        print_digit(x, i);
        printf("Predicted: %d\n", pred);
        printf("Actual: %d\n", true_label);
        if(pred == true_label) {
            printf("Correct! [OK]\n");
            correct++;
        } else {
            printf("Wrong! [X]\n");
        }
        
        // Free memory
        free_tensor(sample);
        free_tensor(w1_out);
        free_tensor(relu_out);
        free_tensor(w2_out);
        free_tensor(output);
    }
    
    printf("\nSummary: Correctly predicted %d out of 10 digits, accuracy = %.1f%%\n", 
           correct, (float)correct * 10);
    
    // Free memory
    free_tensor(x);
    free_tensor(y);
    free_tensor(w1);
    free_tensor(w2);
    
    return 0;
} 