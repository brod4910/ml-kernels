namespace ml::operators::cuda {
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
}