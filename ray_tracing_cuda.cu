#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gputimer.h"
#include "cuda.h"
#include "curand_kernel.h"

#define DOT(X, Y) (X[0]*Y[0] + X[1]*Y[1] + X[2]*Y[2])

__device__ const float L[] = {4, 4, -1};       /* Light source */
__device__ const float C[] = {0, 12, 0};       /* Sphere center */
// __device__ const float R = 6;                  /* Sphere radius */
__device__ const float Wy = 2;                 /* Window y-position */
__device__ const float Wmax = 2;               /* Window dimensions */
__device__ const float magicnum = 36 - 144;

// __device__ inline float SIN(float x) {
//     float xshift = x - M_PI/2;
//     return 1 - xshift*xshift/2 + xshift*xshift*xshift*xshift/24;
// }

// __device__ inline float COS(float x) {
//     float xshift = x - M_PI/2;
//     return -xshift + xshift*xshift*xshift/6 - xshift*xshift*xshift*xshift*xshift/120;
// }

__host__ void writeOutput(void *data, long sz){
    printf("Writing file: sphere.bin\n");
    FILE *file = fopen("sphere.bin", "w");
    fwrite(data, sizeof(float), sz, file);
    fclose(file);
}

__device__ inline void prod(float a, float *X) {
    for (int i = 0; i < 3; i++)
        X[i] *= a;
}

__global__ void simulate_rays(float *Gdev, int *samplesdev, int nrays, int ngrid, int nt) {
    int row, col, samplespriv = 0;
    float V[3], W[3], I[3], N[3], S[3];
    float phi, costheta, sintheta, t, VdotC, disc, NdotS, b;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateXORWOW_t rng;
    curand_init(id, id, 0, &rng);

    for (int z = id; z < nrays; z += nt) {
        do {
            phi = curand_uniform(&rng) * M_PI;
            costheta = curand_uniform(&rng) * 2 - 1;
            sintheta = sqrt(1 - costheta*costheta);
            samplespriv += 2;

            V[0] = sintheta * cos(phi);
            V[1] = sintheta * sin(phi);
            V[2] = costheta;

            for (int i = 0; i < 3; i++) {
                W[i] = (Wy / V[1]) * V[i];
            }

            VdotC = DOT(V, C);

        } while ((disc = VdotC*VdotC + magicnum) < 0 || -Wmax >= W[0] || W[0] >= Wmax || -Wmax >= W[2] || W[2] >= Wmax);

        t = VdotC - sqrt(disc);
        for (int i = 0; i < 3; i++)
            I[i] = t * V[i];
        for (int i = 0; i < 3; i++) {
            N[i] = I[i] - C[i];
            S[i] = L[i] - I[i];
        }
        prod(1/sqrt(DOT(N, N)), N);
        prod(1/sqrt(DOT(S, S)), S);
        NdotS = DOT(N, S);
        b = NdotS > 0 ? NdotS : 0;

        row = ngrid - 1 - floorf((W[0] + Wmax) * ngrid / (2 * Wmax));
        col = floorf((W[2] + Wmax) * ngrid / (2 * Wmax));

        atomicAdd(&Gdev[row*ngrid + col], b);
    }
    atomicAdd(&samplesdev[blockIdx.x], samplespriv);

    return;
}

int main(int argc, char **argv) {
    long long total_samples = 0;
    GpuTimer timer, ktimer;
    int nrays, ngrid, nblocks, ntpb, nt;

    timer.Start();

    if (argc < 5) {
        printf("Usage: ./ray_tracing_gpu <nrays> <ngrid> <nblocks> <ntpb>\n");
        exit(0);
    } else {
        nrays = atoi(argv[1]);
        ngrid = atoi(argv[2]);
        nblocks = atoi(argv[3]);
        ntpb = atoi(argv[4]);
    }

    printf("Parameters: nrays: %d, ngrid: %d, nblocks: %d, threads per block: %d\n", nrays, ngrid, nblocks, ntpb);

    nt = nblocks * ntpb;

    float *G = (float *)malloc(sizeof(float)*ngrid*ngrid);
    float *Gdev;
    cudaMalloc((void **)&Gdev, ngrid*ngrid*sizeof(float));
    cudaMemset((void *)Gdev, 0, ngrid*ngrid*sizeof(float));

    int *samples = (int *)malloc(sizeof(int)*nblocks);
    int *samplesdev;
    cudaMalloc((void **)&samplesdev, sizeof(int)*(nblocks));
    cudaMemset((void *)samplesdev, 0, sizeof(int)*(nblocks));

    ktimer.Start();
    simulate_rays<<<nblocks, ntpb, ntpb>>>(Gdev, samplesdev, nrays, ngrid, nt);
    ktimer.Stop();

    cudaMemcpy(G, Gdev, ngrid*ngrid*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(samples, samplesdev, sizeof(int)*(nblocks), cudaMemcpyDeviceToHost);
    writeOutput(G, ngrid*ngrid);
    for (int i = 0; i < nblocks; i++) {
        total_samples += samples[i];
    }

    printf("RNG samples: %ld\n", total_samples);
    printf("Kernel time = %g ms\n", ktimer.Elapsed());

    free(G);
    free(samples);
    cudaFree(samplesdev);
    cudaFree(Gdev);

    timer.Stop();
    printf("Total time = %g ms\n", timer.Elapsed());
    return 0;
}