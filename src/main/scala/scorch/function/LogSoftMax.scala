package scorch.function

import ns.{Region, Tensor}
import scorch.Variable
import scorch.Function
import torch.cpu.TH

case class LogSoftMax(input: Variable, dim: Option[Long] = None)(
    implicit r: Region)
    extends Function {

  // see torch.nn.functional.py
  // this seems to be deprecated now, so you should provide a dimension
  val dimValue: Long = dim.getOrElse(
    if (input.dim == 0 || input.dim == 1 || input.dim == 3)
      0
    else
      1)

  /*
  THNN_FloatLogSoftMax_updateOutput(SWIGTYPE_p_void state,
                                    THFloatTensor input,
                                    THFloatTensor output,
                                    long dim)
   */
  val output: Tensor = ns.empty
  override def forward(): Variable = {
    TH.THNN_FloatLogSoftMax_updateOutput(null, input, output, dimValue)
    Variable(output, Some(this))
  }

  /*
  THNN_FloatLogSoftMax_updateGradInput(SWIGTYPE_p_void state,
                                       THFloatTensor input,
                                       THFloatTensor gradOutput,
                                       THFloatTensor gradInput,
                                       THFloatTensor output,
                                       long dim)

   */
  override def backward(gradOutput: Variable): Unit = {
    TH.THNN_FloatLogSoftMax_updateGradInput(null,
                                            input,
                                            gradOutput,
                                            input.grad,
                                            output,
                                            dimValue)
    input.backward(input.grad)
  }
}
