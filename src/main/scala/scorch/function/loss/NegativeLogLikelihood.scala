package scorch.function.loss

import ns.Tensor
import scorch.{Function, Variable}
import torch.cpu.{TH, THLongTensor}

case class NegativeLogLikelihood(input: Variable,
                                 target: Variable,
                                 weights: Option[Tensor] = None,
                                 sizeAverage: Boolean = true,
                                 ignoreIndex: Int = -100,
                                 reduce: Boolean = true)
    extends Function {

  val targetAsLong: THLongTensor = ns.floatTensorToLongTensor(target)

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
      targetAsLong,
      output,
      sizeAverage,
      weights.orNull,
      null,
      ignoreIndex,
      reduce
    )

    Variable(output, Some(this))
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

  override def backward(gradOutput: Variable): Unit = {
    TH.THNN_FloatClassNLLCriterion_updateGradInput(
      null,
      input,
      targetAsLong,
      gradOutput,
      input.grad,
      sizeAverage,
      weights.orNull,
      null,
      ignoreIndex,
      reduce
    )
    input.backward(input.grad)
  }
}
