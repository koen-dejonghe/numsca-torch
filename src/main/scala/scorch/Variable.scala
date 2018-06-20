package scorch

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging
import torch.cpu.THFloatTensor

import scala.language.implicitConversions

object Variable {
  def apply(d: Double): Variable = Variable(Tensor(d))
  def apply(d: Double, name: Option[String]): Variable =
    Variable(Tensor(d), name = name)

  implicit def moduleApply[T <: Module](m: T): (Variable) => Variable =
    m.forward
}

case class Variable(data: Tensor,
                    gradFn: Option[Function] = None,
                    name: Option[String] = None)
  extends LazyLogging {

  override def toString: String =
    if (name.isDefined) s"name: ${name.get}, data: $data" else s"data: $data"

  def array: THFloatTensor = data.array

  lazy val grad: Variable =
    Variable(ns.zerosLike(data), name = name.map(n => s"g_$n"))
  def shape: List[Int] = data.shape

  def backward(): Unit = {
    backward(Variable(ns.ones(data.shape)))
  }

  def backward(gradOutput: Variable): Unit = {
    grad.data += gradOutput.data
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  // chain operator
  def ~>(f: (Variable) => Variable): Variable = f(this)


}

