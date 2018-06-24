package scorch

abstract class Optimizer(parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)
}

