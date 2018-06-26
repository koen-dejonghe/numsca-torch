package scorch

import com.typesafe.scalalogging.LazyLogging
import ns.{Region, Tensor}
import torch.cpu.{TH, THFloatTensor}

import scala.language.implicitConversions

object Variable {
  def apply(d: Double)(implicit r: Region): Variable = Variable(Tensor(d.toFloat))
  def apply(d: Double, name: Option[String])(implicit r: Region): Variable =
    Variable(Tensor(d.toFloat), name = name)

  implicit def moduleApply[T <: Module](m: T): (Variable) => Variable =
    m.forward

  implicit def toRawTensor(v: Variable): THFloatTensor = v.array
  implicit def toTensor(v: Variable): Tensor = v.data
}

case class Variable(data: Tensor,
                    gradFn: Option[Function] = None,
                    name: Option[String] = None)(implicit r: Region)
  extends LazyLogging {

  lazy val grad: Variable =
    Variable(ns.zerosLike(data), name = name.map(n => s"g_$n"))

  r.register(() => {
    // logger.debug("disposing of arrays")
    if (gradFn.isDefined) {
      logger.debug("disposing grad")
      grad.data.dispose()
      // grad.array.delete()
      // TH.THFloatTensor_free(grad.array)
      // TH.THFloatStorage_free(grad.array.getStorage)
    }
    logger.debug("disposing")
    data.dispose()
    // data.array.delete()
    // TH.THFloatTensor_free(data.array)
    // TH.THFloatStorage_free(data.array.getStorage)
  })

  override def toString: String =
    if (name.isDefined) s"name: ${name.get}, data: $data" else s"data: $data"

  def array: THFloatTensor = data.array

  def shape: List[Int] = data.shape

  def backward(): Unit = {
    backward(Variable(ns.ones(data.shape)))
  }

//  def backward(): Unit = {
//    backward(this.grad)
//  }

  def backward(gradOutput: Variable): Unit = {
    // grad.data += gradOutput.data // todo: taken care of now by the producing function. Verify.
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  // chain operator
  def ~>(f: (Variable) => Variable): Variable = f(this)


}

