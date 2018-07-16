package scorch.module

import ns.Tensor
import scorch.{Function, Module, Variable}
import torch.cpu.TH

case class BatchNorm(gamma: Variable,
                     beta: Variable,
                     eps: Double = 1e-5,
                     momentum: Double = 0.9)
    extends Module(Seq(gamma, beta)) {

  inTrainingMode = true

  val runningMean: Tensor = ns.zerosLike(gamma.data)
  val runningVar: Tensor = ns.onesLike(gamma.data)

  override def forward(x: Variable): Variable =
    BatchNormFunction(x,
                      eps,
                      momentum,
                      runningMean,
                      runningVar,
                      gamma,
                      beta,
                      inTrainingMode).forward()
}

object BatchNorm {

  def apply(d: Int, eps: Double, momentum: Double): BatchNorm = {
    val gamma = Variable(ns.ones(d))
    val beta = Variable(ns.zeros(d))
    BatchNorm(gamma, beta, eps, momentum)
  }
  def apply(d: Int): BatchNorm = apply(d, eps = 1e-5, momentum = 0.9)
}

case class BatchNormFunction(x: Variable,
                             eps: Double,
                             momentum: Double,
                             runningMean: Tensor,
                             runningVar: Tensor,
                             gamma: Variable,
                             beta: Variable,
                             inTrainingMode: Boolean)
    extends Function {

  /*
  THNN_FloatBatchNormalization_updateOutput (
    SWIGTYPE_p_void state,
    THFloatTensor input,
    THFloatTensor output,
    THFloatTensor weight,
    THFloatTensor bias,
    THFloatTensor running_mean,
    THFloatTensor running_var,
    THFloatTensor save_mean,
    THFloatTensor save_std,
    boolean train,
    double momentum,
    double eps)
   */

// todo not sure what tensor.new() in Lua means
    val saveMean: Tensor = ns.empty
    val saveStd: Tensor = ns.empty
//  val saveMean: Tensor = x.data.copy()
//  val saveStd: Tensor = x.data.copy()


  override def forward(): Variable = {
    val output = ns.empty
    TH.THNN_FloatBatchNormalization_updateOutput(null,
                                                 x,
                                                 output,
                                                 gamma,
                                                 beta,
                                                 runningMean,
                                                 runningVar,
                                                 saveMean,
                                                 saveStd,
                                                 inTrainingMode,
                                                 momentum,
                                                 eps)
    Variable(output, Some(this))
  }

  /*
  THNN_FloatBatchNormalization_backward(SWIGTYPE_p_void state,
    THFloatTensor input,
    THFloatTensor gradOutput,
    THFloatTensor gradInput,
    THFloatTensor gradWeight,
    THFloatTensor gradBias,
    THFloatTensor weight,
    THFloatTensor running_mean,
    THFloatTensor running_var,
    THFloatTensor save_mean,
    THFloatTensor save_std,
    boolean train,
    double scale,
    double eps)
   */

  override def backward(gradOutput: Variable): Unit = {
    TH.THNN_FloatBatchNormalization_backward(null,
                                             x,
                                             gradOutput,
                                             x.grad,
                                             gamma.grad,
                                             beta.grad,
                                             gamma,
                                             runningMean,
                                             runningVar,
                                             saveMean,
                                             saveStd,
                                             inTrainingMode,
                                             momentum,
                                             eps)
    x.backward(x.grad)
  }
}
