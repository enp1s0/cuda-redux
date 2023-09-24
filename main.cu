#include <iostream>

__global__ void redux_kernel(
  std::uint32_t* const out_ptr,
  const std::uint32_t* const in_ptr
  ) {
  const auto in = in_ptr[threadIdx.x];

  constexpr unsigned member_mask = ~0u;

  std::uint32_t out = 0;

  asm(
    R"(
{redux.sync.add.u32 %0, %1, %2;}
)":"=r"(out) : "r"(in), "r"(member_mask)
    );

  out_ptr[threadIdx.x] = out;
}

int main() {
  constexpr std::uint32_t warp_size = 32;

  std::uint32_t *in_ptr, *out_ptr;
  cudaMallocManaged(&in_ptr , sizeof(std::uint32_t) * warp_size);
  cudaMallocManaged(&out_ptr, sizeof(std::uint32_t) * warp_size);

  std::uint32_t ref = 0;
  for (unsigned i = 0; i < warp_size; i++) {
    ref += (in_ptr[i] = i);
  }

  redux_kernel<<<1, warp_size>>>(out_ptr, in_ptr);

  cudaDeviceSynchronize();

  bool ok = true;
  for (unsigned i = 0; i < warp_size; i++) {
    if (out_ptr[i] != ref) {
      std::printf("out[%3u] = %u, ref = %u\n", i, out_ptr[i], ref);
      ok = false;
    }
  }
  std::printf("[result] %s\n", ok ? "ok" : "ng");

  cudaFree(in_ptr);
  cudaFree(out_ptr);
}
