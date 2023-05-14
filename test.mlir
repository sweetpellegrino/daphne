//IR after parsing:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %2 = "daphne.ewAdd"(%0, %1) : (si64, si64) -> si64
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%2, %3, %4) : (si64, i1, i1) -> ()
    %5 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %8 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %10 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %11 = "daphne.cast"(%5) : (si64) -> index
    %12 = "daphne.cast"(%6) : (si64) -> index
    %13 = "daphne.randMatrix"(%11, %12, %7, %8, %9, %10) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    %14 = "daphne.constant"() {value = true} : () -> i1
    %15 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%13, %14, %15) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %16 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %17 = "daphne.constant"() {value = true} : () -> i1
    %18 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%16, %17, %18) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %19 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %20 = "daphne.constant"() {value = true} : () -> i1
    %21 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%19, %20, %21) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %22 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %23 = "daphne.constant"() {value = false} : () -> i1
    %24 = "daphne.matMul"(%13, %22, %23, %23) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    %25 = "daphne.constant"() {value = true} : () -> i1
    %26 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%24, %25, %26) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %27 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %28 = "daphne.constant"() {value = true} : () -> i1
    %29 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%27, %28, %29) : (!daphne.String, i1, i1) -> ()
    %30 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %31 = "daphne.constant"() {value = true} : () -> i1
    %32 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%30, %31, %32) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = 1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.ewAdd
%1 = "daphne.constant"() {value = 2 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.ewAdd
%2 = "daphne.ewAdd"(%0, %1) : (si64, si64) -> si64
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%3 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%4 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%2, %3, %4) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%5 = "daphne.constant"() {value = 2 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.cast
%6 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.cast
%7 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%10 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%11 = "daphne.cast"(%5) : (si64) -> index
//Visiting op 'daphne.cast' with 1 operands:
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%12 = "daphne.cast"(%6) : (si64) -> index
//Visiting op 'daphne.cast' with 1 operands:
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%13 = "daphne.randMatrix"(%11, %12, %7, %8, %9, %10) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.cast'
//  - Operand produced by operation 'daphne.cast'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
%14 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%15 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%13, %14, %15) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%16 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%17 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%18 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%16, %17, %18) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%19 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%20 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%21 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%19, %20, %21) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%22 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.matMul
%23 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.matMul
//    - daphne.matMul
%24 = "daphne.matMul"(%13, %22, %23, %23) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%25 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%26 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%24, %25, %26) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%27 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%28 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%29 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%27, %28, %29) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%30 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%31 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%32 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%30, %31, %32) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = 1 : si64} : () -> si64
  %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
  %2 = "daphne.ewAdd"(%0, %1) : (si64, si64) -> si64
  %3 = "daphne.constant"() {value = true} : () -> i1
  %4 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%2, %3, %4) : (si64, i1, i1) -> ()
  %5 = "daphne.constant"() {value = 2 : si64} : () -> si64
  %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %7 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  %8 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %9 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %10 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %11 = "daphne.cast"(%5) : (si64) -> index
  %12 = "daphne.cast"(%6) : (si64) -> index
  %13 = "daphne.randMatrix"(%11, %12, %7, %8, %9, %10) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  %14 = "daphne.constant"() {value = true} : () -> i1
  %15 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%13, %14, %15) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %16 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  %17 = "daphne.constant"() {value = true} : () -> i1
  %18 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%16, %17, %18) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %19 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  %20 = "daphne.constant"() {value = true} : () -> i1
  %21 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%19, %20, %21) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %22 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  %23 = "daphne.constant"() {value = false} : () -> i1
  %24 = "daphne.matMul"(%13, %22, %23, %23) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
  %25 = "daphne.constant"() {value = true} : () -> i1
  %26 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%24, %25, %26) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %27 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %28 = "daphne.constant"() {value = true} : () -> i1
  %29 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%27, %28, %29) : (!daphne.String, i1, i1) -> ()
  %30 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %31 = "daphne.constant"() {value = true} : () -> i1
  %32 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%30, %31, %32) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %2 = "daphne.ewAdd"(%0, %1) : (si64, si64) -> si64
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%2, %3, %4) : (si64, i1, i1) -> ()
    %5 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %8 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %10 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %11 = "daphne.cast"(%5) : (si64) -> index
    %12 = "daphne.cast"(%6) : (si64) -> index
    %13 = "daphne.randMatrix"(%11, %12, %7, %8, %9, %10) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    %14 = "daphne.constant"() {value = true} : () -> i1
    %15 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%13, %14, %15) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %16 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %17 = "daphne.constant"() {value = true} : () -> i1
    %18 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%16, %17, %18) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %19 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %20 = "daphne.constant"() {value = true} : () -> i1
    %21 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%19, %20, %21) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %22 = "daphne.transpose"(%13) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %23 = "daphne.constant"() {value = false} : () -> i1
    %24 = "daphne.matMul"(%13, %22, %23, %23) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    %25 = "daphne.constant"() {value = true} : () -> i1
    %26 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%24, %25, %26) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %27 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %28 = "daphne.constant"() {value = true} : () -> i1
    %29 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%27, %28, %29) : (!daphne.String, i1, i1) -> ()
    %30 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %31 = "daphne.constant"() {value = true} : () -> i1
    %32 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%30, %31, %32) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after parsing and some simplifications:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%1 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%2 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%3 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%4 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%7 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
"daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - daphne.matMul
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
"daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = true} : () -> i1
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = 3 : index} : () -> index
  %3 = "daphne.constant"() {value = 2 : index} : () -> index
  %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
  %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after SQL parsing:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%1 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%2 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%3 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%4 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%7 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
"daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - daphne.matMul
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
"daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = true} : () -> i1
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = 3 : index} : () -> index
  %3 = "daphne.constant"() {value = 2 : index} : () -> index
  %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
  %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
  "daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
  "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after property inference
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%1 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%2 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%3 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%4 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%7 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
"daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - daphne.matMul
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
"daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = true} : () -> i1
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = 3 : index} : () -> index
  %3 = "daphne.constant"() {value = 2 : index} : () -> index
  %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
  %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
  "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
  "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after type adaptation
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%1 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%2 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%3 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%4 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%7 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
"daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - daphne.matMul
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
"daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = true} : () -> i1
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = 3 : index} : () -> index
  %3 = "daphne.constant"() {value = 2 : index} : () -> index
  %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
  %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
  "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
  "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after vectorization
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%1 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%2 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%3 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%4 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%7 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
"daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - daphne.matMul
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
"daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = true} : () -> i1
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = 3 : index} : () -> index
  %3 = "daphne.constant"() {value = 2 : index} : () -> index
  %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
  %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
  "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
  "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : index} : () -> index
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %7 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    "daphne.print"(%4, %0, %1) : (si64, i1, i1) -> ()
    %11 = "daphne.randMatrix"(%3, %2, %10, %9, %8, %7) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%11, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %12 = "daphne.ewAdd"(%11, %11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%12, %0, %1) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %13 = "daphne.transpose"(%11) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %0, %1) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.matMul"(%11, %11, %1, %0) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %0, %1) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.print"(%6, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %0, %1) : (!daphne.String, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after managing object references
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 2 : index} : () -> index
    %8 = "daphne.constant"() {value = 3 : index} : () -> index
    %9 = "daphne.constant"() {value = false} : () -> i1
    %10 = "daphne.constant"() {value = true} : () -> i1
    %11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
    %12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
    "daphne.print"(%6, %10, %9) : (si64, i1, i1) -> ()
    %13 = "daphne.randMatrix"(%7, %8, %0, %1, %2, %3) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.decRef"(%14) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
    %15 = "daphne.transpose"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%15, %10, %9) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.decRef"(%15) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>) -> ()
    %16 = "daphne.matMul"(%13, %13, %9, %10) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.decRef"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
    "daphne.print"(%16, %10, %9) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.decRef"(%16) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>) -> ()
    "daphne.print"(%4, %10, %9) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %10, %9) : (!daphne.String, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%3 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%6 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%7 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%8 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.randMatrix
%9 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%10 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.matMul
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
//    - daphne.print
%11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.createDaphneContext
%12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
//Visiting op 'daphne.createDaphneContext' with 1 operands:
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has no uses
"daphne.print"(%6, %10, %9) : (si64, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%13 = "daphne.randMatrix"(%7, %8, %0, %1, %2, %3) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.randMatrix' with 6 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 7 uses:
//    - daphne.decRef
//    - daphne.matMul
//    - daphne.matMul
//    - daphne.transpose
//    - daphne.ewAdd
//    - daphne.ewAdd
//    - daphne.print
"daphne.print"(%13, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
%14 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.decRef
//    - daphne.print
"daphne.print"(%14, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.decRef"(%14) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
//Visiting op 'daphne.decRef' with 1 operands:
//  - Operand produced by operation 'daphne.ewAdd'
//Has 0 results:
%15 = "daphne.transpose"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.transpose' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.decRef
//    - daphne.print
"daphne.print"(%15, %10, %9) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.transpose'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.decRef"(%15) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>) -> ()
//Visiting op 'daphne.decRef' with 1 operands:
//  - Operand produced by operation 'daphne.transpose'
//Has 0 results:
%16 = "daphne.matMul"(%13, %13, %9, %10) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.matMul' with 4 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.randMatrix'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.decRef
//    - daphne.print
"daphne.decRef"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
//Visiting op 'daphne.decRef' with 1 operands:
//  - Operand produced by operation 'daphne.randMatrix'
//Has 0 results:
"daphne.print"(%16, %10, %9) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.matMul'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.decRef"(%16) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>) -> ()
//Visiting op 'daphne.decRef' with 1 operands:
//  - Operand produced by operation 'daphne.matMul'
//Has 0 results:
"daphne.print"(%4, %10, %9) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.print"(%5, %10, %9) : (!daphne.String, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//Has 0 results:
"daphne.destroyDaphneContext"() : () -> ()
//Visiting op 'daphne.destroyDaphneContext' with 0 operands:
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  %1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %3 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %7 = "daphne.constant"() {value = 2 : index} : () -> index
  %8 = "daphne.constant"() {value = 3 : index} : () -> index
  %9 = "daphne.constant"() {value = false} : () -> i1
  %10 = "daphne.constant"() {value = true} : () -> i1
  %11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
  %12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
  "daphne.print"(%6, %10, %9) : (si64, i1, i1) -> ()
  %13 = "daphne.randMatrix"(%7, %8, %0, %1, %2, %3) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%13, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  %14 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.print"(%14, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
  "daphne.decRef"(%14) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
  %15 = "daphne.transpose"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
  "daphne.print"(%15, %10, %9) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  "daphne.decRef"(%15) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>) -> ()
  %16 = "daphne.matMul"(%13, %13, %9, %10) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
  "daphne.decRef"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
  "daphne.print"(%16, %10, %9) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
  "daphne.decRef"(%16) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>) -> ()
  "daphne.print"(%4, %10, %9) : (!daphne.String, i1, i1) -> ()
  "daphne.print"(%5, %10, %9) : (!daphne.String, i1, i1) -> ()
  "daphne.destroyDaphneContext"() : () -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 2 : index} : () -> index
    %8 = "daphne.constant"() {value = 3 : index} : () -> index
    %9 = "daphne.constant"() {value = false} : () -> i1
    %10 = "daphne.constant"() {value = true} : () -> i1
    %11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
    %12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
    "daphne.print"(%6, %10, %9) : (si64, i1, i1) -> ()
    %13 = "daphne.randMatrix"(%7, %8, %0, %1, %2, %3) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%13, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %14 = "daphne.ewAdd"(%13, %13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%14, %10, %9) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.decRef"(%14) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
    %15 = "daphne.transpose"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.print"(%15, %10, %9) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.decRef"(%15) : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>) -> ()
    %16 = "daphne.matMul"(%13, %13, %9, %10) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.decRef"(%13) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> ()
    "daphne.print"(%16, %10, %9) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1) -> ()
    "daphne.decRef"(%16) : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>) -> ()
    "daphne.print"(%4, %10, %9) : (!daphne.String, i1, i1) -> ()
    "daphne.print"(%5, %10, %9) : (!daphne.String, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after kernel lowering
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 2 : index} : () -> index
    %8 = "daphne.constant"() {value = 3 : index} : () -> index
    %9 = "daphne.constant"() {value = false} : () -> i1
    %10 = "daphne.constant"() {value = true} : () -> i1
    %11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
    %12 = "daphne.call_kernel"(%11) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    "daphne.call_kernel"(%6, %10, %9, %12) {callee = "_print__int64_t__bool__bool"} : (si64, i1, i1, !daphne.DaphneContext) -> ()
    %13 = "daphne.call_kernel"(%7, %8, %0, %1, %2, %3, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%13, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    %14 = "daphne.call_kernel"(%13, %13, %12) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%14, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%14, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %15 = "daphne.call_kernel"(%13, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%15, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%15, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %16 = "daphne.call_kernel"(%13, %13, %9, %10, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%13, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%4, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%5, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%12) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%3 = "daphne.constant"() {value = -1 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%6 = "daphne.constant"() {value = 3 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%7 = "daphne.constant"() {value = 2 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%8 = "daphne.constant"() {value = 3 : index} : () -> index
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%9 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
%10 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
%11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.call_kernel
%12 = "daphne.call_kernel"(%11) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
//Visiting op 'daphne.call_kernel' with 1 operands:
//  - Operand produced by operation 'daphne.constant'
//Has 1 results:
//  - Result 0 has 16 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
"daphne.call_kernel"(%6, %10, %9, %12) {callee = "_print__int64_t__bool__bool"} : (si64, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
%13 = "daphne.call_kernel"(%7, %8, %0, %1, %2, %3, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.call_kernel' with 7 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 1 results:
//  - Result 0 has 7 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
//    - daphne.call_kernel
"daphne.call_kernel"(%13, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
%14 = "daphne.call_kernel"(%13, %13, %12) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
//Visiting op 'daphne.call_kernel' with 3 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
"daphne.call_kernel"(%14, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%14, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 2 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
%15 = "daphne.call_kernel"(%13, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.call_kernel' with 2 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
"daphne.call_kernel"(%15, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%15, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 2 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
%16 = "daphne.call_kernel"(%13, %13, %9, %10, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
//Visiting op 'daphne.call_kernel' with 5 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - daphne.call_kernel
//    - daphne.call_kernel
"daphne.call_kernel"(%13, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 2 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%16, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%16, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 2 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%4, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%5, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 4 operands:
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.constant'
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.call_kernel"(%12) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
//Visiting op 'daphne.call_kernel' with 1 operands:
//  - Operand produced by operation 'daphne.call_kernel'
//Has 0 results:
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
  %1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
  %2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %3 = "daphne.constant"() {value = -1 : si64} : () -> si64
  %4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
  %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
  %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
  %7 = "daphne.constant"() {value = 2 : index} : () -> index
  %8 = "daphne.constant"() {value = 3 : index} : () -> index
  %9 = "daphne.constant"() {value = false} : () -> i1
  %10 = "daphne.constant"() {value = true} : () -> i1
  %11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
  %12 = "daphne.call_kernel"(%11) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
  "daphne.call_kernel"(%6, %10, %9, %12) {callee = "_print__int64_t__bool__bool"} : (si64, i1, i1, !daphne.DaphneContext) -> ()
  %13 = "daphne.call_kernel"(%7, %8, %0, %1, %2, %3, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.call_kernel"(%13, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
  %14 = "daphne.call_kernel"(%13, %13, %12) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
  "daphne.call_kernel"(%14, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%14, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
  %15 = "daphne.call_kernel"(%13, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
  "daphne.call_kernel"(%15, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%15, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
  %16 = "daphne.call_kernel"(%13, %13, %9, %10, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
  "daphne.call_kernel"(%13, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%16, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%16, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%4, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%5, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
  "daphne.call_kernel"(%12) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
    %2 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = "Hello world!"} : () -> !daphne.String
    %5 = "daphne.constant"() {value = "Bye!"} : () -> !daphne.String
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 2 : index} : () -> index
    %8 = "daphne.constant"() {value = 3 : index} : () -> index
    %9 = "daphne.constant"() {value = false} : () -> i1
    %10 = "daphne.constant"() {value = true} : () -> i1
    %11 = "daphne.constant"() {value = 140722930457888 : ui64} : () -> ui64
    %12 = "daphne.call_kernel"(%11) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    "daphne.call_kernel"(%6, %10, %9, %12) {callee = "_print__int64_t__bool__bool"} : (si64, i1, i1, !daphne.DaphneContext) -> ()
    %13 = "daphne.call_kernel"(%7, %8, %0, %1, %2, %3, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%13, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    %14 = "daphne.call_kernel"(%13, %13, %12) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%14, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%14, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %15 = "daphne.call_kernel"(%13, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> !daphne.Matrix<3x2xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%15, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%15, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<3x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %16 = "daphne.call_kernel"(%13, %13, %9, %10, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%13, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %10, %9, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%4, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%5, %10, %9, %12) {callee = "_print__char__bool__bool"} : (!daphne.String, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%12) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after llvm lowering
module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__char__bool__bool(!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_print__int64_t__bool__bool(i64, i1, i1, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+02 : f64) : f64
    %1 = llvm.mlir.constant(2.000000e+02 : f64) : f64
    %2 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(-1 : si64) : i64
    %4 = llvm.mlir.constant(13 : i64) : i64
    %5 = llvm.alloca %4 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(72 : i8) : i8
    llvm.store %7, %5 : !llvm.ptr<i8>
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.getelementptr %5[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %10 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %10, %9 : !llvm.ptr<i8>
    %11 = llvm.mlir.constant(2 : i64) : i64
    %12 = llvm.getelementptr %5[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %13 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %13, %12 : !llvm.ptr<i8>
    %14 = llvm.mlir.constant(3 : i64) : i64
    %15 = llvm.getelementptr %5[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %16 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %16, %15 : !llvm.ptr<i8>
    %17 = llvm.mlir.constant(4 : i64) : i64
    %18 = llvm.getelementptr %5[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %19 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %19, %18 : !llvm.ptr<i8>
    %20 = llvm.mlir.constant(5 : i64) : i64
    %21 = llvm.getelementptr %5[5] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %22 = llvm.mlir.constant(32 : i8) : i8
    llvm.store %22, %21 : !llvm.ptr<i8>
    %23 = llvm.mlir.constant(6 : i64) : i64
    %24 = llvm.getelementptr %5[6] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %25 = llvm.mlir.constant(119 : i8) : i8
    llvm.store %25, %24 : !llvm.ptr<i8>
    %26 = llvm.mlir.constant(7 : i64) : i64
    %27 = llvm.getelementptr %5[7] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %28 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %28, %27 : !llvm.ptr<i8>
    %29 = llvm.mlir.constant(8 : i64) : i64
    %30 = llvm.getelementptr %5[8] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %31 = llvm.mlir.constant(114 : i8) : i8
    llvm.store %31, %30 : !llvm.ptr<i8>
    %32 = llvm.mlir.constant(9 : i64) : i64
    %33 = llvm.getelementptr %5[9] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %34 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %34, %33 : !llvm.ptr<i8>
    %35 = llvm.mlir.constant(10 : i64) : i64
    %36 = llvm.getelementptr %5[10] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %37 = llvm.mlir.constant(100 : i8) : i8
    llvm.store %37, %36 : !llvm.ptr<i8>
    %38 = llvm.mlir.constant(11 : i64) : i64
    %39 = llvm.getelementptr %5[11] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %40 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %40, %39 : !llvm.ptr<i8>
    %41 = llvm.mlir.constant(12 : i64) : i64
    %42 = llvm.getelementptr %5[12] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %43 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %43, %42 : !llvm.ptr<i8>
    %44 = llvm.mlir.constant(5 : i64) : i64
    %45 = llvm.alloca %44 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %46 = llvm.mlir.constant(0 : i64) : i64
    %47 = llvm.mlir.constant(66 : i8) : i8
    llvm.store %47, %45 : !llvm.ptr<i8>
    %48 = llvm.mlir.constant(1 : i64) : i64
    %49 = llvm.getelementptr %45[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %50 = llvm.mlir.constant(121 : i8) : i8
    llvm.store %50, %49 : !llvm.ptr<i8>
    %51 = llvm.mlir.constant(2 : i64) : i64
    %52 = llvm.getelementptr %45[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %53 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %53, %52 : !llvm.ptr<i8>
    %54 = llvm.mlir.constant(3 : i64) : i64
    %55 = llvm.getelementptr %45[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %56 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %56, %55 : !llvm.ptr<i8>
    %57 = llvm.mlir.constant(4 : i64) : i64
    %58 = llvm.getelementptr %45[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %59 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %59, %58 : !llvm.ptr<i8>
    %60 = llvm.mlir.constant(3 : si64) : i64
    %61 = llvm.mlir.constant(2 : index) : i64
    %62 = llvm.mlir.constant(3 : index) : i64
    %63 = llvm.mlir.constant(false) : i1
    %64 = llvm.mlir.constant(true) : i1
    %65 = llvm.mlir.constant(140722930457888 : ui64) : i64
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.alloca %66 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %68 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %68, %67 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%67, %65) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %69 = llvm.load %67 : !llvm.ptr<ptr<i1>>
    %70 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__int64_t__bool__bool(%60, %64, %63, %69) : (i64, i1, i1, !llvm.ptr<i1>) -> ()
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %73 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%72, %61, %62, %0, %1, %2, %3, %69) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%74, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %76 = llvm.mlir.constant(1 : i64) : i64
    %77 = llvm.alloca %76 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %78 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %78, %77 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%77, %74, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %79 = llvm.load %77 : !llvm.ptr<ptr<i1>>
    %80 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%79, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %81 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%79, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %82 = llvm.mlir.constant(1 : i64) : i64
    %83 = llvm.alloca %82 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %84 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %84, %83 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%83, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %85 = llvm.load %83 : !llvm.ptr<ptr<i1>>
    %86 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%85, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %87 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%85, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %88 = llvm.mlir.constant(1 : i64) : i64
    %89 = llvm.alloca %88 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %90 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %90, %89 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%89, %74, %74, %63, %64, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %91 = llvm.load %89 : !llvm.ptr<ptr<i1>>
    %92 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%74, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %93 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%91, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %94 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%91, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %95 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%5, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %96 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%45, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %97 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%69) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}
llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_print__char__bool__bool(!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_print__int64_t__bool__bool(i64, i1, i1, !llvm.ptr<i1>)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
%0 = llvm.mlir.constant(1.000000e+02 : f64) : f64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%1 = llvm.mlir.constant(2.000000e+02 : f64) : f64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%2 = llvm.mlir.constant(1.000000e+00 : f64) : f64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%3 = llvm.mlir.constant(-1 : si64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%4 = llvm.mlir.constant(13 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%5 = llvm.alloca %4 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 14 uses:
//    - llvm.store
//    - llvm.call
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
%6 = llvm.mlir.constant(0 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%7 = llvm.mlir.constant(72 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %7, %5 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
%8 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%9 = llvm.getelementptr %5[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%10 = llvm.mlir.constant(101 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %10, %9 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%11 = llvm.mlir.constant(2 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%12 = llvm.getelementptr %5[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%13 = llvm.mlir.constant(108 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %13, %12 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%14 = llvm.mlir.constant(3 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%15 = llvm.getelementptr %5[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%16 = llvm.mlir.constant(108 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %16, %15 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%17 = llvm.mlir.constant(4 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%18 = llvm.getelementptr %5[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%19 = llvm.mlir.constant(111 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %19, %18 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%20 = llvm.mlir.constant(5 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%21 = llvm.getelementptr %5[5] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%22 = llvm.mlir.constant(32 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %22, %21 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%23 = llvm.mlir.constant(6 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%24 = llvm.getelementptr %5[6] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%25 = llvm.mlir.constant(119 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %25, %24 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%26 = llvm.mlir.constant(7 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%27 = llvm.getelementptr %5[7] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%28 = llvm.mlir.constant(111 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %28, %27 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%29 = llvm.mlir.constant(8 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%30 = llvm.getelementptr %5[8] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%31 = llvm.mlir.constant(114 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %31, %30 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%32 = llvm.mlir.constant(9 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%33 = llvm.getelementptr %5[9] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%34 = llvm.mlir.constant(108 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %34, %33 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%35 = llvm.mlir.constant(10 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%36 = llvm.getelementptr %5[10] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%37 = llvm.mlir.constant(100 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %37, %36 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%38 = llvm.mlir.constant(11 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%39 = llvm.getelementptr %5[11] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%40 = llvm.mlir.constant(33 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %40, %39 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%41 = llvm.mlir.constant(12 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%42 = llvm.getelementptr %5[12] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%43 = llvm.mlir.constant(0 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %43, %42 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%44 = llvm.mlir.constant(5 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%45 = llvm.alloca %44 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 6 uses:
//    - llvm.store
//    - llvm.call
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
//    - llvm.getelementptr
%46 = llvm.mlir.constant(0 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%47 = llvm.mlir.constant(66 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %47, %45 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
%48 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%49 = llvm.getelementptr %45[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%50 = llvm.mlir.constant(121 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %50, %49 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%51 = llvm.mlir.constant(2 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%52 = llvm.getelementptr %45[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%53 = llvm.mlir.constant(101 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %53, %52 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%54 = llvm.mlir.constant(3 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%55 = llvm.getelementptr %45[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%56 = llvm.mlir.constant(33 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %56, %55 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%57 = llvm.mlir.constant(4 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
%58 = llvm.getelementptr %45[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//Visiting op 'llvm.getelementptr' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
%59 = llvm.mlir.constant(0 : i8) : i8
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %59, %58 : !llvm.ptr<i8>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.getelementptr'
//Has 0 results:
%60 = llvm.mlir.constant(3 : si64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%61 = llvm.mlir.constant(2 : index) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%62 = llvm.mlir.constant(3 : index) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%63 = llvm.mlir.constant(false) : i1
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
%64 = llvm.mlir.constant(true) : i1
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has 8 uses:
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
%65 = llvm.mlir.constant(140722930457888 : ui64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.call
%66 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%67 = llvm.alloca %66 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 3 uses:
//    - llvm.call
//    - llvm.load
//    - llvm.store
%68 = llvm.mlir.null : !llvm.ptr<i1>
//Visiting op 'llvm.mlir.null' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %68, %67 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.null'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
llvm.call @_createDaphneContext__DaphneContext__uint64_t(%67, %65) : (!llvm.ptr<ptr<i1>>, i64) -> ()
//Visiting op 'llvm.call' with 2 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 0 results:
%69 = llvm.load %67 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.load' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has 16 uses:
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
%70 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__int64_t__bool__bool(%60, %64, %63, %69) : (i64, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%71 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 3 uses:
//    - llvm.call
//    - llvm.load
//    - llvm.store
%73 = llvm.mlir.null : !llvm.ptr<i1>
//Visiting op 'llvm.mlir.null' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.null'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%72, %61, %62, %0, %1, %2, %3, %69) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 8 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.load' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has 7 uses:
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
//    - llvm.call
%75 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__DenseMatrix_double__bool__bool(%74, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%76 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%77 = llvm.alloca %76 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 3 uses:
//    - llvm.call
//    - llvm.load
//    - llvm.store
%78 = llvm.mlir.null : !llvm.ptr<i1>
//Visiting op 'llvm.mlir.null' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %78, %77 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.null'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%77, %74, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%79 = llvm.load %77 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.load' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - llvm.call
//    - llvm.call
%80 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__DenseMatrix_double__bool__bool(%79, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%81 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_decRef__Structure(%79, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 2 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%82 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%83 = llvm.alloca %82 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 3 uses:
//    - llvm.call
//    - llvm.load
//    - llvm.store
%84 = llvm.mlir.null : !llvm.ptr<i1>
//Visiting op 'llvm.mlir.null' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %84, %83 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.null'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%83, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 3 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%85 = llvm.load %83 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.load' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - llvm.call
//    - llvm.call
%86 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__DenseMatrix_double__bool__bool(%85, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%87 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_decRef__Structure(%85, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 2 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%88 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.alloca
%89 = llvm.alloca %88 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.alloca' with 1 operands:
//  - Operand produced by operation 'llvm.mlir.constant'
//Has 1 results:
//  - Result 0 has 3 uses:
//    - llvm.call
//    - llvm.load
//    - llvm.store
%90 = llvm.mlir.null : !llvm.ptr<i1>
//Visiting op 'llvm.mlir.null' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - llvm.store
llvm.store %90, %89 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.store' with 2 operands:
//  - Operand produced by operation 'llvm.mlir.null'
//  - Operand produced by operation 'llvm.alloca'
//Has 0 results:
llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%89, %74, %74, %63, %64, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 6 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%91 = llvm.load %89 : !llvm.ptr<ptr<i1>>
//Visiting op 'llvm.load' with 1 operands:
//  - Operand produced by operation 'llvm.alloca'
//Has 1 results:
//  - Result 0 has 2 uses:
//    - llvm.call
//    - llvm.call
%92 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_decRef__Structure(%74, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 2 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%93 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__DenseMatrix_double__bool__bool(%91, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%94 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_decRef__Structure(%91, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 2 operands:
//  - Operand produced by operation 'llvm.load'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%95 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__char__bool__bool(%5, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%96 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_print__char__bool__bool(%45, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 4 operands:
//  - Operand produced by operation 'llvm.alloca'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.mlir.constant'
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
%97 = llvm.mlir.constant(1 : i64) : i64
//Visiting op 'llvm.mlir.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has no uses
llvm.call @_destroyDaphneContext(%69) : (!llvm.ptr<i1>) -> ()
//Visiting op 'llvm.call' with 1 operands:
//  - Operand produced by operation 'llvm.load'
//Has 0 results:
llvm.return
//Visiting op 'llvm.return' with 0 operands:
//Has 0 results:
llvm.func @main() attributes {llvm.emit_c_interface} {
  %0 = llvm.mlir.constant(1.000000e+02 : f64) : f64
  %1 = llvm.mlir.constant(2.000000e+02 : f64) : f64
  %2 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %3 = llvm.mlir.constant(-1 : si64) : i64
  %4 = llvm.mlir.constant(13 : i64) : i64
  %5 = llvm.alloca %4 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
  %6 = llvm.mlir.constant(0 : i64) : i64
  %7 = llvm.mlir.constant(72 : i8) : i8
  llvm.store %7, %5 : !llvm.ptr<i8>
  %8 = llvm.mlir.constant(1 : i64) : i64
  %9 = llvm.getelementptr %5[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %10 = llvm.mlir.constant(101 : i8) : i8
  llvm.store %10, %9 : !llvm.ptr<i8>
  %11 = llvm.mlir.constant(2 : i64) : i64
  %12 = llvm.getelementptr %5[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %13 = llvm.mlir.constant(108 : i8) : i8
  llvm.store %13, %12 : !llvm.ptr<i8>
  %14 = llvm.mlir.constant(3 : i64) : i64
  %15 = llvm.getelementptr %5[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %16 = llvm.mlir.constant(108 : i8) : i8
  llvm.store %16, %15 : !llvm.ptr<i8>
  %17 = llvm.mlir.constant(4 : i64) : i64
  %18 = llvm.getelementptr %5[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %19 = llvm.mlir.constant(111 : i8) : i8
  llvm.store %19, %18 : !llvm.ptr<i8>
  %20 = llvm.mlir.constant(5 : i64) : i64
  %21 = llvm.getelementptr %5[5] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %22 = llvm.mlir.constant(32 : i8) : i8
  llvm.store %22, %21 : !llvm.ptr<i8>
  %23 = llvm.mlir.constant(6 : i64) : i64
  %24 = llvm.getelementptr %5[6] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %25 = llvm.mlir.constant(119 : i8) : i8
  llvm.store %25, %24 : !llvm.ptr<i8>
  %26 = llvm.mlir.constant(7 : i64) : i64
  %27 = llvm.getelementptr %5[7] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %28 = llvm.mlir.constant(111 : i8) : i8
  llvm.store %28, %27 : !llvm.ptr<i8>
  %29 = llvm.mlir.constant(8 : i64) : i64
  %30 = llvm.getelementptr %5[8] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %31 = llvm.mlir.constant(114 : i8) : i8
  llvm.store %31, %30 : !llvm.ptr<i8>
  %32 = llvm.mlir.constant(9 : i64) : i64
  %33 = llvm.getelementptr %5[9] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %34 = llvm.mlir.constant(108 : i8) : i8
  llvm.store %34, %33 : !llvm.ptr<i8>
  %35 = llvm.mlir.constant(10 : i64) : i64
  %36 = llvm.getelementptr %5[10] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %37 = llvm.mlir.constant(100 : i8) : i8
  llvm.store %37, %36 : !llvm.ptr<i8>
  %38 = llvm.mlir.constant(11 : i64) : i64
  %39 = llvm.getelementptr %5[11] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %40 = llvm.mlir.constant(33 : i8) : i8
  llvm.store %40, %39 : !llvm.ptr<i8>
  %41 = llvm.mlir.constant(12 : i64) : i64
  %42 = llvm.getelementptr %5[12] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %43 = llvm.mlir.constant(0 : i8) : i8
  llvm.store %43, %42 : !llvm.ptr<i8>
  %44 = llvm.mlir.constant(5 : i64) : i64
  %45 = llvm.alloca %44 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
  %46 = llvm.mlir.constant(0 : i64) : i64
  %47 = llvm.mlir.constant(66 : i8) : i8
  llvm.store %47, %45 : !llvm.ptr<i8>
  %48 = llvm.mlir.constant(1 : i64) : i64
  %49 = llvm.getelementptr %45[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %50 = llvm.mlir.constant(121 : i8) : i8
  llvm.store %50, %49 : !llvm.ptr<i8>
  %51 = llvm.mlir.constant(2 : i64) : i64
  %52 = llvm.getelementptr %45[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %53 = llvm.mlir.constant(101 : i8) : i8
  llvm.store %53, %52 : !llvm.ptr<i8>
  %54 = llvm.mlir.constant(3 : i64) : i64
  %55 = llvm.getelementptr %45[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %56 = llvm.mlir.constant(33 : i8) : i8
  llvm.store %56, %55 : !llvm.ptr<i8>
  %57 = llvm.mlir.constant(4 : i64) : i64
  %58 = llvm.getelementptr %45[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %59 = llvm.mlir.constant(0 : i8) : i8
  llvm.store %59, %58 : !llvm.ptr<i8>
  %60 = llvm.mlir.constant(3 : si64) : i64
  %61 = llvm.mlir.constant(2 : index) : i64
  %62 = llvm.mlir.constant(3 : index) : i64
  %63 = llvm.mlir.constant(false) : i1
  %64 = llvm.mlir.constant(true) : i1
  %65 = llvm.mlir.constant(140722930457888 : ui64) : i64
  %66 = llvm.mlir.constant(1 : i64) : i64
  %67 = llvm.alloca %66 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
  %68 = llvm.mlir.null : !llvm.ptr<i1>
  llvm.store %68, %67 : !llvm.ptr<ptr<i1>>
  llvm.call @_createDaphneContext__DaphneContext__uint64_t(%67, %65) : (!llvm.ptr<ptr<i1>>, i64) -> ()
  %69 = llvm.load %67 : !llvm.ptr<ptr<i1>>
  %70 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__int64_t__bool__bool(%60, %64, %63, %69) : (i64, i1, i1, !llvm.ptr<i1>) -> ()
  %71 = llvm.mlir.constant(1 : i64) : i64
  %72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
  %73 = llvm.mlir.null : !llvm.ptr<i1>
  llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
  llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%72, %61, %62, %0, %1, %2, %3, %69) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
  %74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
  %75 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__DenseMatrix_double__bool__bool(%74, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
  %76 = llvm.mlir.constant(1 : i64) : i64
  %77 = llvm.alloca %76 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
  %78 = llvm.mlir.null : !llvm.ptr<i1>
  llvm.store %78, %77 : !llvm.ptr<ptr<i1>>
  llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%77, %74, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
  %79 = llvm.load %77 : !llvm.ptr<ptr<i1>>
  %80 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__DenseMatrix_double__bool__bool(%79, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
  %81 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_decRef__Structure(%79, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
  %82 = llvm.mlir.constant(1 : i64) : i64
  %83 = llvm.alloca %82 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
  %84 = llvm.mlir.null : !llvm.ptr<i1>
  llvm.store %84, %83 : !llvm.ptr<ptr<i1>>
  llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%83, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
  %85 = llvm.load %83 : !llvm.ptr<ptr<i1>>
  %86 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__DenseMatrix_double__bool__bool(%85, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
  %87 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_decRef__Structure(%85, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
  %88 = llvm.mlir.constant(1 : i64) : i64
  %89 = llvm.alloca %88 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
  %90 = llvm.mlir.null : !llvm.ptr<i1>
  llvm.store %90, %89 : !llvm.ptr<ptr<i1>>
  llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%89, %74, %74, %63, %64, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
  %91 = llvm.load %89 : !llvm.ptr<ptr<i1>>
  %92 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_decRef__Structure(%74, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
  %93 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__DenseMatrix_double__bool__bool(%91, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
  %94 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_decRef__Structure(%91, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
  %95 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__char__bool__bool(%5, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
  %96 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_print__char__bool__bool(%45, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
  %97 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @_destroyDaphneContext(%69) : (!llvm.ptr<i1>) -> ()
  llvm.return
}
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
llvm.call @main() : () -> ()
//Visiting op 'llvm.call' with 0 operands:
//Has 0 results:
llvm.return
//Visiting op 'llvm.return' with 0 operands:
//Has 0 results:
llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
  llvm.call @main() : () -> ()
  llvm.return
}
//Visiting op 'llvm.func' with 0 operands:
//Has 0 results:
module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__char__bool__bool(!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_print__int64_t__bool__bool(i64, i1, i1, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+02 : f64) : f64
    %1 = llvm.mlir.constant(2.000000e+02 : f64) : f64
    %2 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(-1 : si64) : i64
    %4 = llvm.mlir.constant(13 : i64) : i64
    %5 = llvm.alloca %4 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(72 : i8) : i8
    llvm.store %7, %5 : !llvm.ptr<i8>
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.getelementptr %5[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %10 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %10, %9 : !llvm.ptr<i8>
    %11 = llvm.mlir.constant(2 : i64) : i64
    %12 = llvm.getelementptr %5[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %13 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %13, %12 : !llvm.ptr<i8>
    %14 = llvm.mlir.constant(3 : i64) : i64
    %15 = llvm.getelementptr %5[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %16 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %16, %15 : !llvm.ptr<i8>
    %17 = llvm.mlir.constant(4 : i64) : i64
    %18 = llvm.getelementptr %5[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %19 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %19, %18 : !llvm.ptr<i8>
    %20 = llvm.mlir.constant(5 : i64) : i64
    %21 = llvm.getelementptr %5[5] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %22 = llvm.mlir.constant(32 : i8) : i8
    llvm.store %22, %21 : !llvm.ptr<i8>
    %23 = llvm.mlir.constant(6 : i64) : i64
    %24 = llvm.getelementptr %5[6] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %25 = llvm.mlir.constant(119 : i8) : i8
    llvm.store %25, %24 : !llvm.ptr<i8>
    %26 = llvm.mlir.constant(7 : i64) : i64
    %27 = llvm.getelementptr %5[7] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %28 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %28, %27 : !llvm.ptr<i8>
    %29 = llvm.mlir.constant(8 : i64) : i64
    %30 = llvm.getelementptr %5[8] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %31 = llvm.mlir.constant(114 : i8) : i8
    llvm.store %31, %30 : !llvm.ptr<i8>
    %32 = llvm.mlir.constant(9 : i64) : i64
    %33 = llvm.getelementptr %5[9] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %34 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %34, %33 : !llvm.ptr<i8>
    %35 = llvm.mlir.constant(10 : i64) : i64
    %36 = llvm.getelementptr %5[10] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %37 = llvm.mlir.constant(100 : i8) : i8
    llvm.store %37, %36 : !llvm.ptr<i8>
    %38 = llvm.mlir.constant(11 : i64) : i64
    %39 = llvm.getelementptr %5[11] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %40 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %40, %39 : !llvm.ptr<i8>
    %41 = llvm.mlir.constant(12 : i64) : i64
    %42 = llvm.getelementptr %5[12] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %43 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %43, %42 : !llvm.ptr<i8>
    %44 = llvm.mlir.constant(5 : i64) : i64
    %45 = llvm.alloca %44 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %46 = llvm.mlir.constant(0 : i64) : i64
    %47 = llvm.mlir.constant(66 : i8) : i8
    llvm.store %47, %45 : !llvm.ptr<i8>
    %48 = llvm.mlir.constant(1 : i64) : i64
    %49 = llvm.getelementptr %45[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %50 = llvm.mlir.constant(121 : i8) : i8
    llvm.store %50, %49 : !llvm.ptr<i8>
    %51 = llvm.mlir.constant(2 : i64) : i64
    %52 = llvm.getelementptr %45[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %53 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %53, %52 : !llvm.ptr<i8>
    %54 = llvm.mlir.constant(3 : i64) : i64
    %55 = llvm.getelementptr %45[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %56 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %56, %55 : !llvm.ptr<i8>
    %57 = llvm.mlir.constant(4 : i64) : i64
    %58 = llvm.getelementptr %45[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %59 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %59, %58 : !llvm.ptr<i8>
    %60 = llvm.mlir.constant(3 : si64) : i64
    %61 = llvm.mlir.constant(2 : index) : i64
    %62 = llvm.mlir.constant(3 : index) : i64
    %63 = llvm.mlir.constant(false) : i1
    %64 = llvm.mlir.constant(true) : i1
    %65 = llvm.mlir.constant(140722930457888 : ui64) : i64
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.alloca %66 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %68 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %68, %67 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%67, %65) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %69 = llvm.load %67 : !llvm.ptr<ptr<i1>>
    %70 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__int64_t__bool__bool(%60, %64, %63, %69) : (i64, i1, i1, !llvm.ptr<i1>) -> ()
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %73 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%72, %61, %62, %0, %1, %2, %3, %69) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%74, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %76 = llvm.mlir.constant(1 : i64) : i64
    %77 = llvm.alloca %76 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %78 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %78, %77 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%77, %74, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %79 = llvm.load %77 : !llvm.ptr<ptr<i1>>
    %80 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%79, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %81 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%79, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %82 = llvm.mlir.constant(1 : i64) : i64
    %83 = llvm.alloca %82 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %84 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %84, %83 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%83, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %85 = llvm.load %83 : !llvm.ptr<ptr<i1>>
    %86 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%85, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %87 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%85, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %88 = llvm.mlir.constant(1 : i64) : i64
    %89 = llvm.alloca %88 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %90 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %90, %89 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%89, %74, %74, %63, %64, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %91 = llvm.load %89 : !llvm.ptr<ptr<i1>>
    %92 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%74, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %93 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%91, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %94 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%91, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %95 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%5, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %96 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%45, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %97 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%69) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__char__bool__bool(!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_print__int64_t__bool__bool(i64, i1, i1, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+02 : f64) : f64
    %1 = llvm.mlir.constant(2.000000e+02 : f64) : f64
    %2 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(-1 : si64) : i64
    %4 = llvm.mlir.constant(13 : i64) : i64
    %5 = llvm.alloca %4 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(72 : i8) : i8
    llvm.store %7, %5 : !llvm.ptr<i8>
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.getelementptr %5[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %10 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %10, %9 : !llvm.ptr<i8>
    %11 = llvm.mlir.constant(2 : i64) : i64
    %12 = llvm.getelementptr %5[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %13 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %13, %12 : !llvm.ptr<i8>
    %14 = llvm.mlir.constant(3 : i64) : i64
    %15 = llvm.getelementptr %5[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %16 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %16, %15 : !llvm.ptr<i8>
    %17 = llvm.mlir.constant(4 : i64) : i64
    %18 = llvm.getelementptr %5[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %19 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %19, %18 : !llvm.ptr<i8>
    %20 = llvm.mlir.constant(5 : i64) : i64
    %21 = llvm.getelementptr %5[5] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %22 = llvm.mlir.constant(32 : i8) : i8
    llvm.store %22, %21 : !llvm.ptr<i8>
    %23 = llvm.mlir.constant(6 : i64) : i64
    %24 = llvm.getelementptr %5[6] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %25 = llvm.mlir.constant(119 : i8) : i8
    llvm.store %25, %24 : !llvm.ptr<i8>
    %26 = llvm.mlir.constant(7 : i64) : i64
    %27 = llvm.getelementptr %5[7] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %28 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %28, %27 : !llvm.ptr<i8>
    %29 = llvm.mlir.constant(8 : i64) : i64
    %30 = llvm.getelementptr %5[8] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %31 = llvm.mlir.constant(114 : i8) : i8
    llvm.store %31, %30 : !llvm.ptr<i8>
    %32 = llvm.mlir.constant(9 : i64) : i64
    %33 = llvm.getelementptr %5[9] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %34 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %34, %33 : !llvm.ptr<i8>
    %35 = llvm.mlir.constant(10 : i64) : i64
    %36 = llvm.getelementptr %5[10] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %37 = llvm.mlir.constant(100 : i8) : i8
    llvm.store %37, %36 : !llvm.ptr<i8>
    %38 = llvm.mlir.constant(11 : i64) : i64
    %39 = llvm.getelementptr %5[11] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %40 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %40, %39 : !llvm.ptr<i8>
    %41 = llvm.mlir.constant(12 : i64) : i64
    %42 = llvm.getelementptr %5[12] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %43 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %43, %42 : !llvm.ptr<i8>
    %44 = llvm.mlir.constant(5 : i64) : i64
    %45 = llvm.alloca %44 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %46 = llvm.mlir.constant(0 : i64) : i64
    %47 = llvm.mlir.constant(66 : i8) : i8
    llvm.store %47, %45 : !llvm.ptr<i8>
    %48 = llvm.mlir.constant(1 : i64) : i64
    %49 = llvm.getelementptr %45[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %50 = llvm.mlir.constant(121 : i8) : i8
    llvm.store %50, %49 : !llvm.ptr<i8>
    %51 = llvm.mlir.constant(2 : i64) : i64
    %52 = llvm.getelementptr %45[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %53 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %53, %52 : !llvm.ptr<i8>
    %54 = llvm.mlir.constant(3 : i64) : i64
    %55 = llvm.getelementptr %45[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %56 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %56, %55 : !llvm.ptr<i8>
    %57 = llvm.mlir.constant(4 : i64) : i64
    %58 = llvm.getelementptr %45[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %59 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %59, %58 : !llvm.ptr<i8>
    %60 = llvm.mlir.constant(3 : si64) : i64
    %61 = llvm.mlir.constant(2 : index) : i64
    %62 = llvm.mlir.constant(3 : index) : i64
    %63 = llvm.mlir.constant(false) : i1
    %64 = llvm.mlir.constant(true) : i1
    %65 = llvm.mlir.constant(140722930457888 : ui64) : i64
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.alloca %66 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %68 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %68, %67 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%67, %65) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %69 = llvm.load %67 : !llvm.ptr<ptr<i1>>
    %70 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__int64_t__bool__bool(%60, %64, %63, %69) : (i64, i1, i1, !llvm.ptr<i1>) -> ()
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %73 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%72, %61, %62, %0, %1, %2, %3, %69) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%74, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %76 = llvm.mlir.constant(1 : i64) : i64
    %77 = llvm.alloca %76 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %78 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %78, %77 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%77, %74, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %79 = llvm.load %77 : !llvm.ptr<ptr<i1>>
    %80 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%79, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %81 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%79, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %82 = llvm.mlir.constant(1 : i64) : i64
    %83 = llvm.alloca %82 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %84 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %84, %83 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%83, %74, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %85 = llvm.load %83 : !llvm.ptr<ptr<i1>>
    %86 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%85, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %87 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%85, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %88 = llvm.mlir.constant(1 : i64) : i64
    %89 = llvm.alloca %88 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %90 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %90, %89 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%89, %74, %74, %63, %64, %69) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %91 = llvm.load %89 : !llvm.ptr<ptr<i1>>
    %92 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%74, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %93 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%91, %64, %63, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %94 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%91, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %95 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%5, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %96 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%45, %64, %63, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %97 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%69) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}
