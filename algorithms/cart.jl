#= Classification and Regression Tree =#
mutable struct Node
    data::Matrix
    targets::Vector
    feature_idx::UInt
    threshold::Float64
    lhs::Union{Node, Nothing}
    rhs::Union{Node, Nothing}
end

function classify(x::Vector{Float64}, root::Node)
    if isnothing(root.lhs) && isnothing(root.rhs)
        return sum(root.targets) / length(root.targets) >= 0.5
    elseif x[root.feature_idx] <= root.threshold
        return classify(x, root.lhs)
    else
        return classify(x, root.rhs)
    end
end

function missclassification_error(targets)
    if length(targets) == 0
        return Inf
    end
    p = sum(targets) / length(targets)
    return 1 - p
end

function split(feature::Vector{Float64}, targets::Vector{Int})
    sorted_feature = sort(feature)
    println("Sorted Feature: $sorted_feature")
    best_thresh = sorted_feature[1]
    best_score = Inf
    for threshold in sorted_feature
        mask = feature .<= threshold
        # sum of losses from left and right nodes due to this split
        score = missclassification_error(targets[mask]) + missclassification_error(targets[.!mask])
        println("Score: $score")
        if score < best_score
            best_score = score
            best_thresh = threshold
        end
    end
    println("Best Score: $best_score")
    println("Best Thresh: $best_thresh")
    return best_score, best_thresh
end

function find_split(dataset::Matrix, targets::Vector)
    best_split_score = Inf
    best_threshold = 0.0
    best_feature = 1
    for i = 1:size(dataset, 2)
        feature = dataset[:, i]
        println("Feature: $feature")
        score, threshold = split(feature, targets)
        if score < best_split_score
            best_split_score = score
            best_feature = i
            best_threshold = threshold
        end
    end
    return best_feature, best_threshold
end

function CART(X::Matrix{Float64}, y::Vector{Int64}, min_data_in_leaf::Int64)
    if size(X, 1) <= min_data_in_leaf || sum(y) == 0 || sum(y) == length(y)
        return Node(X, y, 1, 0.0, nothing, nothing)
    else
        feature_idx, threshold = find_split(X, y)
        lhs_mask = X[:, feature_idx] .<= threshold
        rhs_mask = .!lhs_mask
        root = Node(X, y, feature_idx, threshold,
            CART(X[lhs_mask, :], y[lhs_mask], min_data_in_leaf),
            CART(X[rhs_mask, :], y[rhs_mask], min_data_in_leaf)
        )
    end
    return root
end

function print_tree(root::Node)
    queue = [root]
    while length(queue) > 0
        n = pop!(queue)
        println("Node: $n")
        println("Feature: $(n.feature_idx)")
        println("Threshold: $(n.threshold)")
        println("===================================================")
        if !isnothing(n.lhs)
            push!(queue, n.lhs)
        end
        if !isnothing(n.rhs)
            push!(queue, n.rhs)
        end
    end
end