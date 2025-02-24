#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <pcg/pcg_basic.h>

#define DOT(X, Y) (X[0]*Y[0] + X[1]*Y[1] + X[2]*Y[2])

inline double SIN(double x) {
    double xshift = x - M_PI/2;
    return 1 - xshift*xshift/2 + xshift*xshift*xshift*xshift/24;
}

inline double COS(double x) {
    double xshift = x - M_PI/2;
    return -xshift + xshift*xshift*xshift/6 - xshift*xshift*xshift*xshift*xshift/120;
}

void writeOutput(char *filename, double **data, long sz){
    printf("Writing file: %s\n", filename);
    FILE *file = fopen(filename, "w");
    fwrite(*data, sizeof(double), sz, file);
    fclose(file);
}

double **dmatrix(int n) {
    double *data = (double *)malloc(n*n*sizeof(double));
    double **M = (double **)malloc(n*sizeof(double *));
    for (int i = 0; i < n; i++) {
        M[i] = &data[i*n];
    }
    return M;
}

void dmatrix_free(double **M) {
    free(M[0]);
    free(M);
}

/* Multiplies entries of X by a */
inline void prod(double a, double *X) {
    for (int i = 0; i < 3; i++)
        X[i] *= a;
}

int main(int argc, char **argv) {
    uint64_t nrays, samples;
    double **G;
    double st, ft;
    int n, nthreads;

    /* Instance parameters */
    const double L[] = {4, 4, -1};     /* Light source */
    const double C[] = {0, 12, 0};     /* Sphere center */
    const double R = 6;                /* Sphere radius */
    const double Wy = 2;               /* Window y-position */
    const double Wmax = 2;             /* Window dimensions */
    const double magicnum = R*R - DOT(C, C);

    if (argc < 4) {
        printf("Usage: ./ray_tracing <nrays> <ngrid> <nthreads>\n");
        exit(0);
    } else {
        nrays = atoi(argv[1]);
        n = atoi(argv[2]);
        nthreads = atoi(argv[3]);
    }

    omp_set_num_threads(nthreads);
    G = dmatrix(n);
    memset(G[0], 0, n*n*sizeof(double));

    st = omp_get_wtime();

    #pragma omp parallel num_threads(nthreads)
    {
        double** Gpriv = dmatrix(n);
        memset(Gpriv[0], 0, n*n*sizeof(double));
        // int samplespriv = 0;
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);

        int i, j;
        double V[3], W[3], I[3], N[3], S[3];
        double phi, costheta, sintheta, t, VdotC, disc, b;

        #pragma omp for reduction(+ : samples) schedule(dynamic)
        for (int _ = 0; _ < nrays; _++) {
            do {
                phi = (double)pcg32_random_r(&rng) / UINT32_MAX * M_PI;
                costheta = (double)pcg32_random_r(&rng) / UINT32_MAX * 2 - 1;
                sintheta = sqrt(1 - costheta*costheta);
                samples += 2;

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
            b = fmax(0, DOT(N, S));

            i = n - 1 - floor((W[0] + Wmax) * n / (2 * Wmax));
            j = floor((W[2] + Wmax) * n / (2 * Wmax));

            Gpriv[i][j] += b;
        }
        /* Now do data collection */
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    G[i][j] += Gpriv[i][j];
                }
            }
            // samples += samplespriv;
        }

        dmatrix_free(Gpriv);
    }
    
    ft = omp_get_wtime();
    printf("RNG samples: %ld\n", samples);
    printf("Execution time: %f(s)\n", (ft - st));

    writeOutput("sphere.bin", G, n*n);

    dmatrix_free(G);
    return 0;
}