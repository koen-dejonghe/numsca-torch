package scorch.optimizer

import ns.Region
import scorch.{Optimizer, Variable}

case class SGD(parameters: Seq[Variable], lr: Double)
    extends Optimizer(parameters) {
  override def step()(implicit region: Region): Unit =
    parameters.foreach { p =>
      val d = p.grad.data * lr
      p.data -= d
    }
}
