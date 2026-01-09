#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT 12
#define OUTPUT 3
#define LR 0.1f
#define EPOCHS 5000
#define HIDDEN 8

float W1[HIDDEN][INPUT];
float b1[HIDDEN];

float W2[OUTPUT][HIDDEN];
float b2[OUTPUT];

float train_x[][INPUT] = {
    // UP
    {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},

    // MID
    {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0},

    // DOWN
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
};

int train_y[] = {
    0, 0, 0,
    1, 1, 1,
    2, 2, 2};

int TRAIN_N = 9;

float randf()
{
    return ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

float relu(float x)
{
    return x > 0 ? x : 0;
}

void softmax(float *z)
{
    float max = z[0];
    for (int i = 1; i < OUTPUT; i++)
        if (z[i] > max)
            max = z[i];

    float sum = 0;
    for (int i = 0; i < OUTPUT; i++)
    {
        z[i] = expf(z[i] - max);
        sum += z[i];
    }
    for (int i = 0; i < OUTPUT; i++)
        z[i] /= sum;
}

int main()
{
    // w1 and b1 init
    for (int i = 0; i < HIDDEN; i++)
    {
        b1[i] = randf();
        for (int j = 0; j < INPUT; j++)
            W1[i][j] = randf();
    }
    
    // w2 and b2 init
    for (int i = 0; i < OUTPUT; i++)
    {
        b2[i] = 0;
        for (int j = 0; j < HIDDEN; j++)
            W2[i][j] = randf();
    }

    for (int e = 0; e < EPOCHS; e++)
    {
        for (int n = 0; n < TRAIN_N; n++)
        {
            float h[HIDDEN] = {0};
            float z[OUTPUT] = {0};

            for (int i = 0; i < HIDDEN; i++)
            {
                float sum = b1[i];
                for (int j = 0; j < INPUT; j++)
                    sum += W1[i][j] * train_x[n][j];
                h[i] = relu(sum);
            }

            for (int i = 0; i < OUTPUT; i++)
            {
                float sum = b2[i];
                for (int j = 0; j < HIDDEN; j++)
                    sum += W2[i][j] * h[j];
                z[i] = sum;
            }

            softmax(z);

            float dz[OUTPUT];
            float dh[HIDDEN];

            for (int i = 0; i < OUTPUT; i++)
                dz[i] = z[i] - (i == train_y[n]);

            for (int i = 0; i < OUTPUT; i++)
            {
                b2[i] -= LR * dz[i];
                for (int j = 0; j < HIDDEN; j++)
                    W2[i][j] -= LR * dz[i] * h[j];
            }

            for (int j = 0; j < HIDDEN; j++)
            {
                float sum = 0;
                for (int i = 0; i < OUTPUT; i++)
                    sum += W2[i][j] * dz[i];
                dh[j] = h[j] > 0 ? sum : 0;
            }

            for (int j = 0; j < HIDDEN; j++)
            {
                b1[j] -= LR * dh[j];
                for (int k = 0; k < INPUT; k++)
                    W1[j][k] -= LR * dh[j] * train_x[n][k];
            }
        }
    }

    float test[INPUT] = {
        1, 1, 0, 1,
        1, 1, 1, 0,
        0, 0, 0, 1};

    float h[HIDDEN];
    float z[OUTPUT];

    for (int i = 0; i < HIDDEN; i++)
    {
        float sum = b1[i];
        for (int j = 0; j < INPUT; j++)
            sum += W1[i][j] * test[j];
        h[i] = relu(sum);
    }

    // hidden → output
    for (int i = 0; i < OUTPUT; i++)
    {
        float sum = b2[i];
        for (int j = 0; j < HIDDEN; j++)
            sum += W2[i][j] * h[j];
        z[i] = sum;
    }

    softmax(z);

    printf("UP   : %.2f\n", z[0]);
    printf("MID  : %.2f\n", z[1]);
    printf("DOWN : %.2f\n", z[2]);

    return 0;
}