package scorch

import scorch.function.Relu
import scorch.function.loss.MeanSquaredError

trait Function {
  def forward(): Variable
  def backward(gradOutput: Variable): Unit
}

object Function {
  def relu(x: Variable): Variable = Relu(x).forward()
  def mseLoss(x: Variable, y: Variable): Variable = MeanSquaredError(x, y).forward()
}


