package scorch.function

import scorch.{Function, Variable}
import torch.cpu.TH

case class Relu(x: Variable) extends Function {
  override def forward(): Variable = {
    // public static void THNN_FloatLeakyReLU_updateOutput(SWIGTYPE_p_void state, THFloatTensor input, THFloatTensor output, double negval, boolean inplace) {
    val out = ns.empty
    TH.THNN_FloatLeakyReLU_updateOutput(null, x, out, 0.0, false)
    Variable(out, Some(this))
  }

  override def backward(gradOutput: Variable): Unit = {
    // public static void THNN_FloatLeakyReLU_updateGradInput(SWIGTYPE_p_void state, THFloatTensor input, THFloatTensor gradOutput, THFloatTensor gradInput, double negval, boolean inplace) {
    TH.THNN_FloatLeakyReLU_updateGradInput(null,
      x,
      gradOutput,
      x.grad,
      0.0,
      false)
    x.backward(x.grad)
  }
}

