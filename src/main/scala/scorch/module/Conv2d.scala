package scorch.module

import ns.Tensor
import scorch.{Module, Variable}
import torch.cpu.TH

case class Conv2d(weight: Variable,
                  bias: Variable,
                  fInput: Variable,
                  kW: Int,
                  kH: Int,
                  dW: Int,
                  dH: Int,
                  padW: Int,
                  padH: Int,
                  scale: Double)
    extends Module(Seq(weight, bias, fInput)) {

  override def forward(x: Variable): Variable =
    Conv2dFunction(x, weight, bias, fInput, kW, kH, dW, dH, padW, padH, scale)
      .forward()
}

object Conv2d {

  /**
    * Applies a 2D convolution over an input image composed of several input planes.
    * The input tensor in forward(input) is expected to be a 3D tensor (nInputPlane x height x width).
    * @param nInputPlane The number of expected input planes in the image given into forward()
    * @param nOutputPlane  The number of output planes the convolution layer will produce.
    * @param kW The kernel width of the convolution
    * @param kH The kernel height of the convolution
    * @param dW The step of the convolution in the width dimension. Default is 1
    * @param dH The step of the convolution in the height dimension. Default is 1
    * @param padW Padding width
    * @param padH Padding height
    * @return A Conv2 object
    */
  def apply(nInputPlane: Int,
            nOutputPlane: Int,
            kW: Int,
            kH: Int,
            dW: Int,
            dH: Int,
            padW: Int,
            padH: Int,
            scale: Double): Conv2d = {
    val w = Variable(ns.randn(nOutputPlane, nInputPlane, kW, kH))
    val b = Variable(ns.zeros(nOutputPlane))
    val fInput = Variable(w.data.copy())
    Conv2d(w, b, fInput, kW, kH, dW, dH, padW, padH, scale)
  }

  def apply(nInputPlane: Int, nOutputPlane: Int, k: Int): Conv2d =
    apply(nInputPlane, nOutputPlane, k, k, 1, 1, 0, 0, 1.0)

}

case class Conv2dFunction(input: Variable,
                          weight: Variable,
                          bias: Variable,
                          fInput: Variable,
                          kW: Int,
                          kH: Int,
                          dW: Int,
                          dH: Int,
                          padW: Int,
                          padH: Int,
                          scale: Double)
    extends scorch.Function {

  val output: Tensor = ns.empty
  /*
    THNN_FloatSpatialConvolutionMM_updateOutput(SWIGTYPE_p_void state
      THFloatTensor input
      THFloatTensor output
      THFloatTensor weight
      THFloatTensor bias
      THFloatTensor finput
      THFloatTensor fgradInput
      int kW
      int kH
      int dW
      int dH
      int padW
      int padH)
   */

  override def forward(): Variable = {
    TH.THNN_FloatSpatialConvolutionMM_updateOutput(null,
                                                   input,
                                                   output,
                                                   weight,
                                                   bias,
                                                   fInput,
                                                   fInput.grad,
                                                   kW,
                                                   kH,
                                                   dW,
                                                   dH,
                                                   padW,
                                                   padH)

    Variable(output, Some(this))

  }

  override def backward(gradOutput: Variable): Unit = {

    /*
      THNN_FloatSpatialConvolutionMM_updateGradInput(SWIGTYPE_p_void state
         THFloatTensor input
         THFloatTensor gradOutput
         THFloatTensor gradInput
         THFloatTensor weight
         THFloatTensor finput
         THFloatTensor fgradInput
         int kW
         int kH
         int dW
         int dH
         int padW
         int padH) {
     */

    TH.THNN_FloatSpatialConvolutionMM_updateGradInput(
      null,
      input,
      gradOutput,
      input.grad,
      weight,
      fInput,
      fInput.grad,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH
    )

    /*
      THNN_FloatSpatialConvolutionMM_accGradParameters(SWIGTYPE_p_void state
        THFloatTensor input
        THFloatTensor gradOutput
        THFloatTensor gradWeight
        THFloatTensor gradBias
        THFloatTensor finput
        THFloatTensor fgradInput
        int kW
        int kH
        int dW
        int dH
        int padW
        int padH
        double scale) {
     */

    TH.THNN_FloatSpatialConvolutionMM_accGradParameters(null,
                                                        input,
                                                        gradOutput,
                                                        weight.grad,
                                                        bias.grad,
                                                        fInput,
                                                        fInput.grad,
                                                        kW,
                                                        kH,
                                                        dW,
                                                        dH,
                                                        padW,
                                                        padH,
                                                        scale)

  }
}
