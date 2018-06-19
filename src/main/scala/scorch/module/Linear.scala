package scorch.module

import botkop.{numsca => ns}
import botkop.numsca.Tensor
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
                                       x.payload,
                                       out.payload,
                                       weights.payload,
                                       bias.payload,
                                       buffer.payload)
      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      // public static void THNN_FloatLinear_updateGradInput(SWIGTYPE_p_void state, THFloatTensor input, THFloatTensor gradOutput, THFloatTensor gradInput, THFloatTensor weight) {
      TH.THNN_FloatLinear_updateGradInput(null, x.payload, gradOutput.payload, ???, weights.payload)
    }

  }
}
