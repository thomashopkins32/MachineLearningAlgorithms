using FLux
using MLDatasets

function conv2d(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Int = 1)
    W = rand(Float32, in_channels, out_channels, kernel_size[1], kernel_size[2])
    b = rand(Float32, in_channels, out_channels)
    function forward(x::Array{AbstractFloat})
        # x has shape (28, 28, in_channels, batch_size)
        dim1 = size(x, 1)
        dim2 = size(x, 2)
        k1 = kernel_size[1]
        k2 = kernel_size[2]
        out_dim1 = floor((dim1 - k1) / stride)
        out_dim2 = floor((dim2 - k2) / stride)
        res = zeros(Float32, (out_dim1, out_dim2, out_channels, size(x, 4)))
        for i = 1:in_channels, j = 1:out_channels
            # iterate over patches of the image using kernel size and stride
            out1 = 1
            for s1 = 1:dim1:stride
                out2 = 1
                for s2 = 1:dim2:stride
                    # sum over element-wise product and store result
                    res[out1, out2, j, :] = sum(W[i, j] .* x[s1:s1 + k1, s2:s2 + k2, i, :]; dims=(1, 2)) .+ b[i, j]
                    out2 += 1
                end
                out1 += 1
            end
        end
        return res
    end
    return forward
end

