#pragma once

void sgemm_cuda(int M, int N, int K, float alpha, float beta, int num_runs = 1000);