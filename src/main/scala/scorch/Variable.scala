package scorch

import com.typesafe.scalalogging.LazyLogging
import ns.Tensor
import torch.cpu.THFloatTensor

import scala.language.implicitConversions

object Variable {
  def apply(d: Double): Variable = Variable(Tensor(d.toFloat))
  def apply(d: Double, name: Option[String]): Variable =
    Variable(Tensor(d.toFloat), name = name)

  implicit def moduleApply[T <: Module](m: T): (Variable) => Variable =
    m.forward

  implicit def toRawTensor(v: Variable): THFloatTensor = v.array
  implicit def toTensor(v: Variable): Tensor = v.data
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
    // grad.data += gradOutput.data // todo: taken care of now by the producing function. Verify.
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  // chain operator
  def ~>(f: (Variable) => Variable): Variable = f(this)


}

