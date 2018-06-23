package scorch.function.loss

import ns.Tensor
import scorch.{Function, Variable}
import torch.cpu.TH

case class NegativeLogLikelyhood(input: Variable,
                                 target: Variable,
                                 weights: Option[Tensor] = None,
                                 sizeAverage: Boolean = true,
                                 ignoreIndex: Int = -100,
                                 reduce: Boolean = true)
    extends Function {

  /*
  THNN_FloatClassNLLCriterion_updateOutput(
    SWIGTYPE_p_void state,
    THFloatTensor input,
    THLongTensor target,
    THFloatTensor output,
    boolean sizeAverage,
    THFloatTensor weights,
    THFloatTensor total_weight,
    long ignore_index,
    boolean reduce
  )
   */

  override def forward(): Variable = {
    val output = ns.empty

    TH.THNN_FloatClassNLLCriterion_updateOutput(
      null,
      input,
      target,
      output,
      sizeAverage,
      weights.orNull,
      null,
      ignoreIndex,
      reduce
    )
  }

  /*
  THNN_FloatClassNLLCriterion_updateGradInput(
    SWIGTYPE_p_void state,
    THFloatTensor input,
    THLongTensor target,
    THFloatTensor gradOutput,
    THFloatTensor gradInput,
    boolean sizeAverage,
    THFloatTensor weights,
    THFloatTensor total_weight,
    long ignore_index,
    boolean reduce
  )
   */

  override def backward(gradOutput: Variable): Unit = ???
}
