function setup_best_gpu()
    devs = CUDA.devices()
    mem_free = Float64[]

    for (i, dev) in enumerate(devs)
        CUDA.device!(dev)
        CUDA.zeros(1)

        free, total = CUDA.memory_info()
        push!(mem_free, free)

        println("Device $(i-1): ",
            round(free/1_048_576, digits=2),
            " MB free")
    end

    best_idx = argmax(mem_free)
    best_dev = collect(devs)[best_idx]

    CUDA.device!(best_dev)
    println("Selected GPU: $(best_idx-1)")
end