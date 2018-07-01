package scorch.module

import com.typesafe.scalalogging.LazyLogging
import scorch.{Module, Variable}

case class Parallel(module: Module, parallelism: Int)
    extends Module {
  import scorch.module.Parallel._
  override def forward(x: Variable): Variable =
    Parallelize(x, module, parallelism).forward()
}

object Parallel {

  case class Parallelize(x: Variable,
                         module: Module,
                         parallelism: Int)
      extends scorch.Function
      with LazyLogging {
    import ns._

    val batchSize: Int = x.shape.head
    val chunkSize: Int = Math.max(batchSize / parallelism, 1)

    private val fromTos = (0 until batchSize)
      .sliding(chunkSize, chunkSize)
      .map(s => (s.head, s.last + 1))
      .toArray
      .par

    lazy val (xs, predictions) =
      fromTos.map {
        case (first, last) =>
          val cx = Variable(x.data(first :> last).copy())
          val r = module(cx)
          (cx, r)
      }.unzip

    override def forward(): Variable = {
      val pas = predictions.map(_.data).seq
      val cc = ns.concatenate(pas, 0)
      Variable(cc, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      predictions.zip(fromTos).foreach {
        case (v, (first, last)) =>
          val g = Variable(gradOutput.data(first :> last).copy())
          v.backward(g)
      }

      val gradient = Variable(
        ns.concatenate(xs.map(_.grad.data).seq, 0))
      x.backward(gradient)
    }
  }
}
