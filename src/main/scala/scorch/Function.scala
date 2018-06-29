package scorch

import ns.Tensor
import scorch.function._
import scorch.function.loss._

trait Function {
  def forward(): Variable
  def backward(gradOutput: Variable): Unit
}

object Function {
  def relu(x: Variable): Variable = LeakyRelu(x).forward()

  def leakyRelu(x: Variable, negVal: Double = 0.0): Variable =
    LeakyRelu(x, negVal).forward()

  def logSoftMax(x: Variable, dim: Option[Long] = None): Variable =
    LogSoftMax(x, dim).forward()

  /* === loss functions ============================================================== */
  def mseLoss(x: Variable,
              y: Variable,
              sizeAverage: Boolean = true,
              reduce: Boolean = true): Variable =
    MeanSquaredError(x, y, sizeAverage, reduce).forward()

  def nll(x: Variable,
          y: Variable,
          weights: Option[Tensor] = None,
          sizeAverage: Boolean = true,
          ignoreIndex: Int = -100,
          reduce: Boolean = true): Variable =
    NegativeLogLikelihood(x, y, weights, sizeAverage, ignoreIndex, reduce)
      .forward()

  def crossEntropy(x: Variable,
                   y: Variable,
                   weights: Option[Tensor] = None,
                   sizeAverage: Boolean = true,
                   ignoreIndex: Int = -100,
                   reduce: Boolean = true): Variable =
    nll(logSoftMax(x, Some(1)), y, weights, sizeAverage, ignoreIndex, reduce)
}
