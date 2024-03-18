#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <mpi.h>
#define PI 3.14159265358979323846
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))

// constant L and C vectors for the GPU
__constant__ float L_d_const[3];
__constant__ float C_d_const[3];

// OpenMP version for CPU
__host__ void ray_tracing_omp(float ***G, float *cnt, float *L, float *C, int N_rays, int G_size, float Wy, float W_max, float R, int nthreads)
{
    unsigned int seed = 420;
#pragma omp parallel for num_threads(nthreads) default(none) shared(W_max, Wy, N_rays, G, C, R, L, G_size, cnt) private(seed) schedule(static)
    for (int v_idx = 0; v_idx < N_rays; v_idx++)
    {
        float b, phi, cos_theta, sin_theta, root_temp, t, VC;
        float V[3], W[3], mag;
        int i_idx, j_idx;
        do
        {
            // sample phi and cos theta till it intersects the window
            phi = (float)rand_r(&seed) / RAND_MAX * PI;
            cos_theta = (float)rand_r(&seed) / RAND_MAX * 2.0f - 1.0f;
            sin_theta = sqrt(1.0f - cos_theta * cos_theta);

            V[0] = sin_theta * cos(phi);
            V[1] = sin_theta * sin(phi);
            V[2] = cos_theta;

            W[0] = Wy / V[1] * V[0];
            W[1] = Wy / V[1] * V[1];
            W[2] = Wy / V[1] * V[2];

            VC = V[0] * C[0] + V[1] * C[1] + V[2] * C[2];

            root_temp = VC * VC + R * R - (C[0] * C[0] + C[1] * C[1] + C[2] * C[2]);

            // Commenteed out as this is very expensive for normal time runs, uncomment to display number of generated rays
            // #pragma omp atomic
            // (*cnt)++;

        } while (W[0] * W[0] >= W_max * W_max || W[2] * W[2] >= W_max * W_max || root_temp <= 0 || W[1] <= 0);

        t = VC - sqrt(root_temp);
        float I[3] = {t * V[0], t * V[1], t * V[2]};
        float I_C[3] = {I[0] - C[0], I[1] - C[1], I[2] - C[2]};
        float L_I[3] = {L[0] - I[0], L[1] - I[1], L[2] - I[2]};

        mag = sqrt(I_C[0] * I_C[0] + I_C[1] * I_C[1] + I_C[2] * I_C[2]);
        float Norm[3] = {I_C[0] / mag, I_C[1] / mag, I_C[2] / mag};

        mag = sqrt(L_I[0] * L_I[0] + L_I[1] * L_I[1] + L_I[2] * L_I[2]);
        float S[3] = {L_I[0] / mag, L_I[1] / mag, L_I[2] / mag};

        b = MAX(0.0f, S[0] * Norm[0] + S[1] * Norm[1] + S[2] * Norm[2]);

        i_idx = (W_max - W[0]) * G_size * 0.5f / W_max;
        j_idx = (W_max - W[2]) * G_size * 0.5f / W_max;

        // update the threads's local copy of the grid
        G[omp_get_thread_num()][i_idx][j_idx] += b;
    }
}

__global__ void ray_tracing_cuda(float *G, float *cnt, curandState *state, int N_rays, int G_size, float Wy, float W_max, float R)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x, i_idx, j_idx;
    float b, phi, cos_theta, sin_theta, root_temp, t, VC, V[3], W[3], mag;
    // extracting the initialized PRG for cuda thread id
    curandState localState = state[id];
    // grid stride loop
    for (int i = id; i < N_rays; i += blockDim.x * gridDim.x)
    {
        do
        {
            phi = (float)curand_uniform(&localState) * PI;
            cos_theta = (float)curand_uniform(&localState) * 2.0f - 1.0f;
            sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            V[0] = sin_theta * cosf(phi);
            V[1] = sin_theta * sinf(phi);
            V[2] = cos_theta;

            W[0] = Wy / V[1] * V[0];
            W[1] = Wy / V[1] * V[1];
            W[2] = Wy / V[1] * V[2];

            VC = V[0] * C_d_const[0] + V[1] * C_d_const[1] + V[2] * C_d_const[2];

            VC = V[0] * C_d_const[0] + V[1] * C_d_const[1] + V[2] * C_d_const[2];

            root_temp = VC * VC + R * R - (C_d_const[0] * C_d_const[0] + C_d_const[1] * C_d_const[1] + C_d_const[2] * C_d_const[2]);
            
            // Commenteed out as this is very expensive for normal time runs, uncomment to display number of generated rays
            // atomicAdd((float *)cnt, 1.0f);

        } while (W[0] * W[0] >= W_max * W_max || W[2] * W[2] >= W_max * W_max || root_temp <= 0 || W[1] <= 0);

        t = VC - sqrt(root_temp);
        float I[3] = {t * V[0], t * V[1], t * V[2]};
        float I_C[3] = {I[0] - C_d_const[0], I[1] - C_d_const[1], I[2] - C_d_const[2]};
        float L_I[3] = {L_d_const[0] - I[0], L_d_const[1] - I[1], L_d_const[2] - I[2]};

        mag = sqrt(I_C[0] * I_C[0] + I_C[1] * I_C[1] + I_C[2] * I_C[2]);
        float Norm[3] = {I_C[0] / mag, I_C[1] / mag, I_C[2] / mag};

        mag = sqrt(L_I[0] * L_I[0] + L_I[1] * L_I[1] + L_I[2] * L_I[2]);
        float S[3] = {L_I[0] / mag, L_I[1] / mag, L_I[2] / mag};

        b = MAX(0.0f, S[0] * Norm[0] + S[1] * Norm[1] + S[2] * Norm[2]);

        i_idx = (W_max - W[0]) * G_size * 0.5f / W_max;
        j_idx = (W_max - W[2]) * G_size * 0.5f / W_max;

        atomicAdd((float *)&G[i_idx * G_size + j_idx], b);
    }
}

// initialize the curand object before the ray computation begins as curand_init is very expensive
__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(id * 4238811, id, 0, &state[id]);
}

int main(int argc, char **argv)
{
    float Wy, W_max, R;
    int nthreads = omp_get_max_threads(), NTPB = 512, G_size = 1000, N_rays = 1e9, run_type=2, nblocks=(1<<16)-1;
    int rank=0,size=1;
    printf("\n");

    N_rays = atoi(argv[1]);
    G_size = atoi(argv[2]);
    nblocks = atoi(argv[3]);
    NTPB = atoi(argv[4]);
    if(argc>5)
        run_type = atoi(argv[5]);
    if(argc>6)
        nthreads = atoi(argv[6]);
    
    // if mpi run
    MPI_Status stat;
    if(run_type==3){
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("rank %d size %d\n",rank,size);
    }

    FILE *file;
    float *C, *L;
    C = (float *)malloc(3 * sizeof(float));
    L = (float *)malloc(3 * sizeof(float));
    C[0] = 0.0f;
    C[1] = 12.0f;
    C[2] = 0.0f;
    L[0] = 4.0f;
    L[1] = 4.0f;
    L[2] = -1.0f;
    Wy = 2.0f;
    W_max = 2.0f;
    R = 6.0f;

    if(run_type!=3 || rank==0){
        printf("C = {%0.2lf %0.2lf %0.2lf}\n", C[0], C[1], C[2]);
        printf("L = {%0.2lf %0.2lf %0.2lf}\n", L[0], L[1], L[2]);
        printf("Wy = %0.2lf W_max = %0.2lf, R = %0.2lf, N_rays = %e, G_size = %d\n", Wy, W_max, R, (float)N_rays, G_size);
    }
    
    // Convert left hand 3D system to right hand 3D system
    L[2] *= -1.0f;
    C[2] *= -1.0f;

    // total time counters
    float total_time;
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total, 0);

    // initialize grid for the image
    float **G = (float **)malloc(G_size * sizeof(float *));
    G[0] = (float *)malloc(G_size * G_size * sizeof(float));
    
    for (int i = 1; i < G_size; i++)
    {
        G[i] = G[0] + i * G_size;
    }

    if(run_type==0 || run_type==1){
        printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CPU~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
        // set all grid cells to zero
        for (int i = 0; i < G_size * G_size; i++)
            G[0][i] = 0.0f;

        // array of 2d grids of each thread
        float ***G_parent;
        G_parent = (float ***)malloc(nthreads*sizeof(float **));
        for(int i=0;i<nthreads;i++){
            G_parent[i] = (float **)malloc(G_size * sizeof(float *));
            G_parent[i][0] = (float *)malloc(G_size * G_size * sizeof(float));
            for (int j = 1; j < G_size; j++){
                G_parent[i][j] = G_parent[i][0] + j * G_size;
                for(int k=0;k<G_size;k++)
                    G_parent[i][j][k] = 0.0f;
            }
        }

        float t1, t2;
        float cnt=0.0;
        printf("nthreads = %d\n", nthreads);
        printf("\nrunning OMP version...\n");

        t1 = omp_get_wtime();
        ray_tracing_omp(G_parent, &cnt, L, C, N_rays, G_size, Wy, W_max, R, nthreads);
        t2 = omp_get_wtime();

        printf("\nOMP Time = %0.3lf seconds \tcnt = %e\n", t2 - t1, cnt);

        // commbine the grids from each thread
        for(int i=0;i<G_size;i++)
            for(int j=0;j<G_size;j++)
                for(int k=0;k<nthreads;k++)
                    G[i][j]+=G_parent[k][i][j];

        // write final image to a file
        file = fopen("omp-image.dat", "w");
        fprintf(file, "%d %d %d %d %d %s\n", G_size, N_rays, nthreads, -1, -1, "float");
        for (int i = 0; i < G_size; i++)
        {
            for (int j = 0; j < G_size; j++)
            {
                fprintf(file, "%lf ", G[i][j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }

    if(run_type==0 || run_type>=2){
        // if non-MPI run or rank 0 in MPI run
        if(run_type!=3 || rank==0){
            printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GPU~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
            printf("nblocks               : %d\nnum_threads_per_block : %d\n", nblocks, NTPB);
            printf("Total GPU threads     : %e\n", (float) nblocks * NTPB);
            printf("Total GPU malloc size : %lf GBs\n",(float)(G_size * G_size * sizeof(float)+nblocks * NTPB * sizeof(curandState))/1.0e9);
            printf("Rays per thread       : %lf rays\n",((float) N_rays)/((float) nblocks * NTPB));    
        }
        if(run_type==3)
            cudaSetDevice(rank);
        // reset all grid cells to zero
        for (int i = 0; i < G_size * G_size; i++)
            G[0][i] = 0.0f;

        float *G_d;
        float *cnt_d,cnt;
        cnt=0.0f;
        cudaEvent_t start, stop; /* kernel timers */
        float time;
        float *G_recv;
        curandState *devStates;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // nblocks = MIN((N_rays + NTPB - 1) / NTPB, (((float) MAX_GPU_MEMORY_IN_GBs*1.0e9-(float)(G_size * G_size * sizeof(float)))/((float)NTPB * sizeof(curandState))));
        
        // malloc and send arrays to device
        assert(cudaMalloc((void **)&G_d, G_size * G_size * sizeof(float)) == cudaSuccess);
        assert(cudaMalloc((void **)&cnt_d, 1 * sizeof(float)) == cudaSuccess);
        assert(cudaMalloc((void **)&devStates, nblocks * NTPB * sizeof(curandState)) == cudaSuccess);

        assert(cudaMemcpy(G_d, G[0], G_size * G_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemcpy(cnt_d, &cnt, 1 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        
        assert(cudaMemcpyToSymbol(L_d_const, L, 3 * sizeof(float)) == cudaSuccess);
        assert(cudaMemcpyToSymbol(C_d_const, C, 3 * sizeof(float)) == cudaSuccess);
        setup_kernel<<<nblocks, NTPB>>>(devStates);
        
        printf("\nrunning CUDA version on GPU %d...\n",rank);
        cudaEventRecord(start, 0);
        
        ray_tracing_cuda<<<nblocks, NTPB>>>(G_d, cnt_d, devStates, N_rays/size, G_size, Wy, W_max, R);
    
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        if(run_type==3){
            // in MPI run, send from rank 1 to rank 0
            if(rank==0){
                G_recv = (float*)malloc(G_size*G_size*sizeof(float));
                MPI_Recv(G_recv, G_size*G_size, MPI_FLOAT, 1, 99, MPI_COMM_WORLD, &stat);
            }
            else{
                assert(cudaMemcpy(G[0], G_d, G_size * G_size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        
                MPI_Send(G[0], G_size * G_size, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
            }
        }

        assert(cudaMemcpy(G[0], G_d, G_size * G_size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(cudaMemcpy(&cnt, cnt_d, 1 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        
        printf("\nCUDA Time (GPU %d) = %lf seconds\t cnt = %e\n", rank, time / 1000., cnt);
        // combine the grids from different ranks
        if(run_type==3 && rank==0){
            for(int i=0;i<G_size*G_size;i++)
                G[0][i] += G_recv[i];
        }

        if(rank==0){
            // write final image to a file
            file = fopen("cuda-image.dat", "w");
            fprintf(file, "%d %d %d %d %d %s\n", G_size, N_rays, -1, nblocks, NTPB, "float");
            for (int i = 0; i < G_size; i++)
            {
                for (int j = 0; j < G_size; j++)
                {
                    fprintf(file, "%lf ", G[i][j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
        // free gpu allocated memory
        cudaFree(G_d);
        cudaFree(cnt_d);
        cudaFree(devStates);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    //free host allocated memory
    free(G[0]);
    free(G);
    free(C);
    free(L);
    // finish mpi run
    if(run_type==3)
        MPI_Finalize();
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_time, start_total, stop_total);
    // print total time on rank 0
    if(rank==0){
        printf("\nTotal Time = %lf seconds\n", total_time / 1000.0f);
        printf("\n");
    }
    return 0;
}