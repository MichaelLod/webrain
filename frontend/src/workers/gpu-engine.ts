import shaderCode from "./shaders/matmul.wgsl";

export class GPUEngine {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  async init(): Promise<{ gpu: string; supported: boolean }> {
    if (!navigator.gpu) {
      return { gpu: "WebGPU not supported", supported: false };
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { gpu: "No GPU adapter found", supported: false };
    }
    this.device = await adapter.requestDevice();
    const info = adapter.info;

    const shaderModule = this.device.createShaderModule({ code: shaderCode });
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    return {
      gpu: info.description || info.vendor || "Unknown GPU",
      supported: true,
    };
  }

  async matmul(
    a: Float32Array,
    b: Float32Array,
    M: number,
    K: number,
    N: number
  ): Promise<Float32Array> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error("GPU not initialized");
    }

    const bufA = this.device.createBuffer({
      size: a.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufB = this.device.createBuffer({
      size: b.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufC = this.device.createBuffer({
      size: M * N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = new Uint32Array([M, K, N]);
    const bufParams = this.device.createBuffer({
      size: params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(bufA, 0, a.buffer);
    this.device.queue.writeBuffer(bufB, 0, b.buffer);
    this.device.queue.writeBuffer(bufParams, 0, params.buffer);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufParams } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
    pass.end();

    const readBuf = this.device.createBuffer({
      size: M * N * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(bufC, 0, readBuf, 0, M * N * 4);
    this.device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    bufA.destroy();
    bufB.destroy();
    bufC.destroy();
    bufParams.destroy();
    readBuf.destroy();

    return result;
  }

  destroy() {
    this.device?.destroy();
    this.device = null;
  }
}
