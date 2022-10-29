function conv2d(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Int = 1)
    W = rand(Float32, in_channels, out_channels, kernel_size[1], kernel_size[2])
    b = rand(Float32, in_channels, out_channels)

    function forward(x::Array{<:AbstractFloat, 4})
        # x has shape (dim1, dim2, in_channels, batch_size)
        dim1 = size(x, 1)
        dim2 = size(x, 2)
        k1 = kernel_size[1]
        k2 = kernel_size[2]
        out_dim1 = floor(Int, (dim1 - k1) / stride) + 1
        out_dim2 = floor(Int, (dim2 - k2) / stride) + 1
        res = zeros(Float32, (out_dim1, out_dim2, out_channels, size(x, 4)))
        for i = 1:in_channels, j = 1:out_channels
            # iterate over patches of the image using kernel size and stride
            out1 = 1
            for s1 = 1:stride:dim1 - k1 + 1
                out2 = 1
                for s2 = 1:stride:dim2 - k2 + 1
                    # sum over element-wise product and store result
                    res[out1, out2, j, :] = sum(W[i, j, :, :] .* x[s1:s1 + k1 - 1, s2:s2 + k2 - 1, i, :]; dims=(1, 2)) .+ b[i, j]
                    out2 += 1
                end
                out1 += 1
            end
        end
        return res
    end

    return forward
end


println("VERIFYING RESULTS")
println("===============================================")

bs = 1
color_channels = 1
dim1 = 5
dim2 = 5
kernel = (3, 3)
stride = 1
println("Params:\n\tBatch Size: $bs\n\tChannels: $color_channels\n\tDims: ($dim1, $dim2)\n\tKernel: $kernel\n\tStride: $stride")
input = Array{Float32, 2}([1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25])
input = reshape(input, (5, 5, 1, 1))
conv_layer = conv2d(color_channels, 1, kernel; stride = stride)
println("Input shape: $(size(input))")
println("Input type: $(typeof(input))")
println("Input: $(input)")
println("Conv W: $(conv_layer.W)")
println("Conv b: $(conv_layer.b)")
output = conv_layer(input)
println("Output shape: $(size(output))")
println("Output: $(output)")

println("\nBIGGER EXAMPLE")
println("===============================================")

bs = 100
color_channels = 3
dim1 = 64
dim2 = 64
kernel = (5, 5)
stride = 2
println("Params:\n\tBatch Size: $bs\n\tChannels: $color_channels\n\tDims: ($dim1, $dim2)\n\tKernel: $kernel\n\tStride: $stride")
input = rand(Float32, dim1, dim2, color_channels, bs)
conv_layer = conv2d(color_channels, 10, kernel; stride = stride)
println("Input shape: $(size(input))")
println("Input type: $(typeof(input))")
output = conv_layer(input)
println("Output shape: $(size(output))")