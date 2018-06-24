package scorch.optimizer

import scorch.{Optimizer, Variable}

case class SGD(parameters: Seq[Variable], lr: Double)
    extends Optimizer(parameters) {
  override def step(): Unit =
    parameters.foreach { p =>
      p.data -= p.grad.data * lr
    }
}
