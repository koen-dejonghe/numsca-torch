package scorch.optimizer

import ns.Tensor
import scorch.{Optimizer, Variable}

// https://arxiv.org/pdf/1711.05101.pdf

class SGDW(parameters: Seq[Variable],
           alpha: Double, // learning rate
           beta: Double = 0, // momentum
           w: Double = 0 // weight decay
) extends Optimizer(parameters) {

  var t = 0
  val ms: Seq[Tensor] = parameters.map(p => ns.zerosLike(p))

  override def step(): Unit = {
    t += 1
    parameters.zip(ms).foreach {
      case (x, m) =>
        // line 8
        m *= beta
        m += x.grad * alpha * t
        // line 9
        x -= m + (x * t * w)
    }
  }

}
