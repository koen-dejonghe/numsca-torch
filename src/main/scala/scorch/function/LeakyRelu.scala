package scorch.function

import scorch.{Function, Variable}
import torch.cpu.TH

case class LeakyRelu(x: Variable, negVal: Double = 0.0) extends Function {

  /*
  THNN_FloatLeakyReLU_updateOutput(SWIGTYPE_p_void state,
                                   THFloatTensor input,
                                   THFloatTensor output,
                                   double negval,
                                   boolean inplace)
   */

  override def forward(): Variable = {
    val out = ns.empty
    TH.THNN_FloatLeakyReLU_updateOutput(null, x, out, negVal, false)
    Variable(out, Some(this))
  }

  /*
  THNN_FloatLeakyReLU_updateGradInput(SWIGTYPE_p_void state,
                                      THFloatTensor input,
                                      THFloatTensor gradOutput,
                                      THFloatTensor gradInput,
                                      double negval,
                                      boolean inplace)
   */
  override def backward(gradOutput: Variable): Unit = {
    TH.THNN_FloatLeakyReLU_updateGradInput(null,
                                           x,
                                           gradOutput,
                                           x.grad,
                                           negVal,
                                           false)

    x.backward(x.grad)
  }
}
