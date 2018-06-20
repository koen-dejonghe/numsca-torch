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

  case class LinearFunction(x: Variable, weights: Variable, bias: Variable)
      extends Function {

    val out: Tensor = ns.empty
    val buffer: Tensor = ns.empty

    override def forward(): Variable = {
      TH.THNN_FloatLinear_updateOutput(null,
                                       x.array,
                                       out.array,
                                       weights.array,
                                       bias.array,
                                       buffer.array)
      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {

      TH.THNN_FloatLinear_updateGradInput(null,
                                          x.array,
                                          gradOutput.array,
                                          x.grad.array,
                                          weights.array)

      TH.THNN_FloatLinear_accGradParameters(null,
                                            x.array,
                                            gradOutput.array,
                                            x.grad.array,
                                            weights.array,
                                            bias.array,
                                            weights.grad.array,
                                            bias.grad.array,
                                            buffer.array,
                                            1.0)
    }

  }
}
