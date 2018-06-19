package scorch

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import torch.cpu.TH

abstract class BaseModule(localParameters: Seq[Variable] = Nil) {

  // by default, obtain submodules through introspection
  lazy val subModules: Seq[BaseModule] =
    this.getClass.getDeclaredFields.flatMap { f =>
      f setAccessible true
      f.get(this) match {
        case module: BaseModule => Some(module)
        case _                  => None
      }
    }

  def parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  def gradients: Seq[Variable] = parameters.map(_.grad)

  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)

  /*
  Pytorch way of solving distinction between training and test mode is by using a mutable variable.
  Perhaps there is a better way.
   */
  var inTrainingMode: Boolean = false

  /*
  Sets the module in training mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def train(mode: Boolean = true): Unit = {
    this.inTrainingMode = mode
    subModules.foreach(_.train(mode))
  }

  /*
  Sets the module in evaluation mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def eval(): Unit = train(false)

}

abstract class Module(localParameters: Seq[Variable] = Nil)
    extends BaseModule(localParameters)
    with LazyLogging {
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}


