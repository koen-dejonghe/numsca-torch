package torch.cpu

import java.util.concurrent.atomic.AtomicLong

import com.typesafe.scalalogging.LazyLogging

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.language.postfixOps

object NaiveMemoryManagement extends App with LazyLogging {

  logger.info("*** starting sequential **************************")
  sequential()
//  logger.info("*** starting parallel ****************************")
//  parallel()

  def sequential(): Unit = {
    val t3 = MyTensor.zeros(100, 100) // this one will only get garbage collected at the end of the program

    for (i <- 1 to 10000000) {
      MyTensor.zeros(3000, 3000) // these will get GC'ed as soon as as System.gc() is called
      Thread.sleep(1)
    }

    logger.info("DONE")
    logger.info(t3.cPtr.toString)
    logger.info(t3.payload.toString)
    logger.info(TH.THFloatTensor_desc(t3.payload).getStr) // this should still work
    logger.info(TH.THFloatTensor_get2d(t3.payload, 10, 10).toString)
  }

  def parallel(): Unit = {

    import scala.concurrent.ExecutionContext.Implicits.global

    val t3 = MyTensor.zeros(100, 100) // this one will only get garbage collected at the end of the program

    val futures = Future.sequence {
      (1 to 100).map { _ =>
        Future {
          MyTensor.zeros(3000, 3000) // these will get GC'ed as soon as as System.gc() is called
          Thread.sleep(1)
        }
      }
    }

    Await.result(futures, 10 seconds)

    logger.info("DONE")
    logger.info(t3.cPtr.toString)
    logger.info(t3.payload.toString)
    logger.info(TH.THFloatTensor_desc(t3.payload).getStr) // this should still work
    logger.info(TH.THFloatTensor_get2d(t3.payload, 10, 10).toString)
  }

}


case class FloatTensor private(p: Long, t: THFloatTensor) {

  def free() = {
    THJNI.THFloatTensor_free(p, t)
  }

}
object FloatTensor {

  def create(): FloatTensor = {
    val p = THJNI.THFloatTensor_new
    val t = new THFloatTensor(p, false)
    FloatTensor(p, t)
  }
}

case class MyTensor private (payload: THFloatTensor,
                             cPtr: Long,
                             size: Long) extends LazyLogging {
  override def finalize(): Unit = {
    THJNI.THFloatTensor_free(cPtr, payload)
    // TH.THFloatTensor_free(payload)
    // payload.delete()
    val memSize = MyTensor.memoryWaterMark.addAndGet(-size)
    logger.info(s"freeing $cPtr (mem = $memSize)")
  }
}

object MyTensor extends LazyLogging {

  val threshold: Long = 2L * 1024L * 1024L * 1024L // 2 GB

  val memoryWaterMark = new AtomicLong(0)

  def memCheck(size: Long): Unit =
    if (memoryWaterMark.addAndGet(size) > threshold) {
      System.gc()
    }

  def zeros(d1: Long, d2: Long): MyTensor = {
    val tensor = makeTensorOfZeros(d1, d2)
    logger.info(s"creating ${tensor.cPtr}")
    memCheck(tensor.size)
    tensor
  }

  // boiler plate to create a Torch tensor of floats
  def makeTensorOfZeros(d1: Long, d2: Long): MyTensor = {
    val size: THLongStorage = TH.THLongStorage_newWithSize2(d1, d2)
    // val cPtr = THJNI.THFloatTensor_newWithSize2d(d1, d2)
    val cPtr = THJNI.THFloatTensor_new
    val t = new THFloatTensor(cPtr, false)
    // val t = new THFloatTensor(cPtr, true)

    // val t = TH.THFloatTensor_new()
    // val t = new THFloatTensor()

    TH.THFloatTensor_zeros(t, size)
    MyTensor(t, cPtr, d1 * d2 * 4) // float = 4 bytes
//    MyTensor(t, 0, d1 * d2 * 4) // float = 4 bytes
  }

}
