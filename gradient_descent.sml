exception InvalidDimensions
open Vector
open Array2

fun sum_vec (v : real vector) : real = 
  foldl (op +) 0.0 v

fun subtract ((v, u) : real vector * real vector) : real vector = 
  mapi (fn (i, v_i) => v_i - Vector.sub (u, i)) v

fun dot ((v, u) : real vector * real vector) : real = 
  foldli (fn (i, v_i, z) => v_i * Vector.sub (u, i) + z) 0.0 v

fun elem_square (v : real vector) : real vector = 
  map (fn v_i => v_i * v_i) v

fun transpose (A : real array) : real array =
  tabulate ColMajor (nCols A, nRows A, fn (r, c) => sub(A, c, r)) 

fun mult_scalar_vec ((a, v) : real * real vector) : real vector = 
  map (fn v_i => a * v_i) v

(* mult_arr_vec : real array * real vector -> real vector
 * REQUIRES : columns of A = len of B
 * ENSURES : mult_arr_vec (A, v) evaluates to the inner product of A and v
 *)
fun mult_arr_vec ((A, v) : real array * real vector) : real vector = 
  let
    val err = if (nCols A <> length v) then raise InvalidDimensions else ();
  in
    Vector.tabulate (nRows A, fn (i) => dot(row (A, i), v))
  end

(* cost : real array * real vector -> real vector -> real
 * REQUIRES : rows X = length Y, cols X = length params
 * ENSURES cost (X, Y) params evaluates the mean squared error
 *)
fun cost ((X, Y) : real array * real vector) params =
  let
    val n = real (length Y)
    val H = mult_arr_vec (X, params)
  in
    (0.5 * n) * (sum_vec (elem_square (subtract (H, Y))))
  end

(* gradient_descent : real array * real vector * real vector * real * int * real list
 *                 -> real list * real vector 
 * REQUIRES : rows X = length Y, cols X = length params
 * ENSURES : history and optimal parameters after training for n_iters
 *)
fun gradient_descent (_, _, params, _ , 0, history) = (history, params)
  | gradient_descent (X, Y, params, lr, n_iters, history) = 
  let
    val n = real (Vector.length Y)
    val H = mult_arr_vec (X, params)
    val gradient = mult_scalar_vec(lr / n, mult_arr_vec (transpose X, subtract(H, Y)))
    val new_params = subtract (params, gradient)
    val new_cost = cost (X, Y) new_params
  in
    gradient_descent (X, Y, new_params, lr, n_iters-1, new_cost::history)
  end

val X = Array2.fromList [
[2.0, 10.0],
[1.0, 9.0] ,
[1.0, 5.0] ,
[3.0, 7.0] ,
[3.0, 10.0] ]

val Y = Vector.fromList [
1000.0,
800.0,
500.0,
900.0,
1200.0 ]

val params = Vector.fromList [0.0, 0.0]
val n_iters = 10000
val lr = 0.01

val (history, optimal_params) = gradient_descent (X, Y, params, lr, n_iters, [])

(* fun get_mean_std_vecs X = 
  let
    open Vector
    open Array2
    val get_mean = fn v => ((sum_vec v) / (real (length v)))
    val mean_vec = Vector.tabulate (nCols X, fn i => get_mean (column (X, i)))
    val get_variance = fn (v, mean) => (sum_vec (elem_square (Vector.map (fn v_i => v_i - mean) v))) / (real (length v))
    val get_stdev = Math.sqrt o get_variance
    val std_vec = Vector.tabulate (nCols X, fn i => get_stdev (column (X, i), Vector.sub (mean_vec, i))) 
  in
    (mean_vec, std_vec)
  end *)

(* fun scale_by (mean_vec, std_vec) X = 
  let
    open Array2
    val scale_element_by = fn (r, c) => (Array2.sub(X, r, c) - Vector.sub(mean_vec, c))/(Vector.sub(std_vec, c))
  in
    Array2.tabulate Array2.ColMajor (Array2.nRows X, Array2.nCols X, scale_element_by) 
  end *)

(* val scale = scale_by (get_mean_std_vecs X)
val X = scale X *)