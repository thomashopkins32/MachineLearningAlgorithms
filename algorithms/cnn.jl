

function conv2d(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Int = 1)
    W = rand(Float32, in_channels, out_channels, kernel_size, kernel_size)
    b = rand(Float32, in_channels, out_channels)
    function forward(x::Array{AbstractFloat})
        # x has shape (28, 28, in_channels, batch_size)
        dim1 = size(x, 1)
        dim2 = size(x, 2)
        # TODO: determine shape of result and initialize result array
        # should be something like (TBD, TBD, out_channels, batch_size)
        res = nothing
        for i = 1:in_channels, j = 1:out_channels
            # iterate over patches of the image using kernel size and stride
            for s1 = 1:dim1:stride, s2 = 1:dim2:stride
                # compute element-wise product with filter (W) and patch
                res = W[i, j] .* x[s1:kernel_size, s2:kernel_size, :]
                # add bias to result
                res .+= b[i, j]
            end
        end
    end
    return forward
end