using LinearAlgebra

function set_diagonal(M, val)
    X = M
    cols = size(M, 1)
    rows = size(M, 2)
    for i in 1:cols, j in 1:rows
        if (i == j)
            X[i, j] = val
        end
    end

    return X
end

function gen_sym_adj_mat(n)
    A = rand(0.0:1.0, n, n)
    A = A - tril(A) + triu(A)'
    A = set_diagonal(A, 1)
    return A
end

const num_of_layers = 3
const matrix_dimensions = 4
# generate network
N = Matrix[]
append!(N, [gen_sym_adj_mat(matrix_dimensions) for i in 1:num_of_layers])
for matrix in N
    display(matrix)
    println()
end

