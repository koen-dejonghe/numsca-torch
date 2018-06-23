package scorch.function.loss

import ns.Tensor
import scorch.{Function, Variable}
import torch.cpu.TH

case class MeanSquaredError(x: Variable,
                            y: Variable,
                            sizeAverage: Boolean = true,
                            reduce: Boolean = true)
    extends Function {

  override def forward(): Variable = {

    /*
    THNN_FloatMSECriterion_updateOutput(state: SWIGTYPE_p_void,
                                        input: THFloatTensor,
                                        target: THFloatTensor,
                                        output: THFloatTensor,
                                        sizeAverage: Boolean,
                                        reduce: Boolean)
     */

    val output: Tensor = ns.empty
    TH.THNN_FloatMSECriterion_updateOutput(null,
                                           x,
                                           y,
                                           output,
                                           sizeAverage,
                                           reduce)
    Variable(output, gradFn = Some(this))
  }

  override def backward(gradOutput: Variable): Unit = {
    TH.THNN_FloatMSECriterion_updateGradInput(null,
                                              x,
                                              y,
                                              gradOutput,
                                              x.grad,
                                              sizeAverage,
                                              reduce)
    x.backward(x.grad)
  }
}
