package scorch.module

import ns.Tensor
import scorch._
import torch.cpu.TH

case class Linear(weights: Variable, bias: Variable)
    extends Module(Seq(weights, bias)) {

  import Linear._

  override def forward(x: Variable): Variable =
    LinearFunction(x, weights, bias).forward()
}

object Linear {

  def apply(inFeatures: Int, outFeatures: Int): Linear = {
    val w: Tensor = ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
    val weights = Variable(w)
    val b: Tensor = ns.zeros(outFeatures)
    val bias = Variable(b)
    Linear(weights, bias)
  }

  case class LinearFunction(x: Variable, weights: Variable, bias: Variable)
      extends Function {

    val out: Tensor = ns.empty
    val buffer: Tensor = ns.empty

    override def forward(): Variable = {
      TH.THNN_FloatLinear_updateOutput(null, x, out, weights, bias, buffer)
      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {

      TH.THNN_FloatLinear_updateGradInput(null, x, gradOutput, x.grad, weights)

      TH.THNN_FloatLinear_accGradParameters(null,
                                            x,
                                            gradOutput,
                                            x.grad,
                                            weights,
                                            bias,
                                            weights.grad,
                                            bias.grad,
                                            buffer,
                                            1)

      x.backward(x.grad)
    }
  }
}
