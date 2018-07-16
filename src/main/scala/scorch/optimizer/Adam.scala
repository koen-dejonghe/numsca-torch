package scorch.optimizer

import ns.Tensor
import scorch.{Optimizer, Variable}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.language.postfixOps

case class Adam(parameters: Seq[Variable],
                lr: Double,
                beta1: Double = 0.9,
                beta2: Double = 0.999,
                epsilon: Double = 1e-8)
    extends Optimizer(parameters) {

  val extParameters: Seq[(Variable, Tensor, Tensor)] = parameters.map { p =>
    (p, ns.zerosLike(p), ns.zerosLike(p))
  }

  var t = 0

  override def step(): Unit = {
    t += 1
    Future.sequence {
      extParameters.map {
        case (p, m, v) =>
          nextStep(p, m, v)
      }
    }
    // not blocking: maybe not very scientific,
    // but in practice it seems to works fine, and it is a lot faster
  }

  def nextStep(p: Variable, m: Tensor, v: Tensor) = Future {
    val x = p.data
    val dx = p.grad.data

    m *= beta1
    m += (1 - beta1) * dx
    val mt = m / (1 - math.pow(beta1, t))

    v *= beta2
    v += (1 - beta2) * ns.square(dx)
    val vt = v / (1 - math.pow(beta2, t))

    x -= lr * mt / (ns.sqrt(vt) + epsilon)
  }

}
