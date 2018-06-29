package scorch.function.loss

import ns.{LongTensor, Tensor}
import scorch.{Function, Variable}
import torch.cpu.TH

import scala.language.implicitConversions

case class NegativeLogLikelihood(input: Variable,
                                 target: Variable,
                                 weights: Option[Tensor] = None,
                                 sizeAverage: Boolean = true,
                                 ignoreIndex: Int = -100,
                                 reduce: Boolean = true)
    extends Function {

  // cast target as long tensor
  val targetAsLongTensor: LongTensor = LongTensor(target)
  println("casted!!!!!!!!!!!!!!") // todo fix this

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

  val totalWeight: Tensor = ns.empty

  override def forward(): Variable = {
    val output = ns.empty

    TH.THNN_FloatClassNLLCriterion_updateOutput(
      null,
      input,
      targetAsLongTensor,
      output,
      sizeAverage,
      weights.map(_.array).orNull,
      totalWeight,
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
      targetAsLongTensor,
      gradOutput,
      input.grad,
      sizeAverage,
      weights.map(_.array).orNull,
      totalWeight,
      ignoreIndex,
      reduce
    )

    input.backward(input.grad)
  }

}
