package scorch

import ns.Region

abstract class Optimizer(parameters: Seq[Variable]) {
  def step()(implicit r: Region): Unit
  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)
}

