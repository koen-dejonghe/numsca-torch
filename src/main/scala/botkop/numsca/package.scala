package botkop

import scala.language.implicitConversions

package object numsca {

  case class NumscaRange(from: Int, to: Option[Int])

  def :>(end: Int) = NumscaRange(0, Some(end))
  def :> = NumscaRange(0, None)

  implicit class NumscaInt(i: Int) {
    def :>(end: Int) = NumscaRange(i, Some(end))
    def :> = NumscaRange(i, None)
  }

  implicit def intToNumscaRange(i: Int): NumscaRange =
    NumscaRange(i, Some(i + 1))

  case class NumscaRanges(rs: Seq[NumscaRange])

}
