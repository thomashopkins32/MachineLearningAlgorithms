#= Classification and Regression Tree =#
mutable struct Node
    data::Matrix
    targets::Vector
    rule::Function
    lhs::Node
    rhs::Node
end

function classify(n::Node)
    return mean(n.targets) >= 0.5
end

function split(feature::Vector, targets::Vector)
    # TODO: Implement split threshold search
end

function find_split(dataset::Matrix, targets::Vector)
    best_split_score = Inf
    best_split_func = nothing
    for feature in transpose(dataset) 
        score, func = split(feature, targets)
        if func !== nothing && score < best_split_score
            best_split_score = score
            best_split_func = func
        end
    end
    return best_split_func
end

function CART(X::Matrix, y::Vector, num_nodes::UInt, max_nodes::UInt)
    if num_nodes >= max_nodes
        return nothing
    end
    split_func = find_split(X, y)
    lhs_mask = split_func(X)
    rhs_mask = ~split_func(X)
    root = Node(X, y, split_func,
        CART(X[lhs_mask, :], y[lhs_mask], num_nodes + 2, max_nodes),
        CART(X[rhs_mask, :], y[rhs_mask], num_nodes + 2, max_nodes)
    )
    return root
end